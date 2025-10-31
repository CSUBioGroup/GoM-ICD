
from src.train_load import *
from src.gap_rnn_moe_model import *
from src.args_parse import *
from src.vocab_all import *
from src.process import *
from src.evaluation import all_metrics

from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AdamW, get_scheduler, AutoConfig
from transformers import AutoTokenizer, RobertaModel
from tqdm.autonotebook import tqdm
import random
import logging
import math
import torch
import datetime

logger = logging.getLogger(__name__)

# set the random seed if needed, disable by default
set_random_seed(random_seed=42)

def main():
    args = create_args_parser()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    raw_datasets = load_data(args.data_dir)
    print(raw_datasets)
    vocab = Vocab(args, raw_datasets)
    label_to_id = vocab.label_to_id
    num_labels = len(label_to_id)

    print(len(vocab.label_dict['train']))
    print(len(vocab.label_dict['valid']))
    print(len(vocab.label_dict['test']))
    remove_columns = raw_datasets["train"].column_names

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    if "roberta" == args.model_type:
        model = Roberta_model.from_pretrained(args.model_name_or_path, config=config, args=args, vocab=vocab)

    def getitem(examples):
        label_list = []
        texts = ((examples["text"],))#or Text
        result = tokenizer(*texts, padding=False, max_length=args.max_seq_length, truncation=True,
                           add_special_tokens=True)
        for labels in examples["target"]:  #or Full_Labels 不同处理
            label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split(',')])
        result["label_ids"] = label_list
        return result

    def data_collator_train(features):
        batch = dict()

        if args.dataEnhance:
            for i in range(len(features)):
                len_fea = int(len(features[i]['input_ids'][1:-1]))
                if random.random() < args.dataEnhanceRatio / 2:
                    features[i]['input_ids'][1:-1] = torch.tensor(features[i]['input_ids'])[1:-1][
                        np.random.permutation(len_fea)].tolist()
                if random.random() < args.dataEnhanceRatio:
                    features[i]['input_ids'][1:-1] = torch.tensor(features[i]['input_ids'])[1:-1][
                        range(len_fea)[::-1]].tolist()

        max_length = max([len(f["input_ids"]) for f in features])

        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size
        list_input_ids = [
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]
        
        original_input_order = list(range(max_length))

        if args.gap_type==0:
            batch["input_ids"] = torch.tensor(list_input_ids).contiguous().view((len(features), -1, args.chunk_size))
            batch["inverse_input_list"]=original_input_order

        elif args.gap_type==1:
            chunk_num=max_length // args.chunk_size
            gap_list = []
            for sublist in list_input_ids:
                sublist_chunks = [sublist[i::chunk_num] for i in range(chunk_num)]
                gap_list.append(sublist_chunks)
            batch["input_ids"]=torch.tensor(gap_list)
            original_input_list = [original_input_order[i::chunk_num] for i in range(chunk_num)]
            original_input_list=sum(original_input_list, [])
            inverse_indices = {index: i for i, index in enumerate(original_input_list)}
            inverse_indices_list = [inverse_indices[i] for i in range(max_length)]
            batch["inverse_input_list"]=inverse_indices_list

        elif args.gap_type == 2:
            chunk_num=max_length // args.chunk_size
            connect_token=args.connect_token
            connect_gap=args.chunk_size//connect_token
            gap_list = []
            for sublist in list_input_ids:
                sublist_chunks = [sublist[i:i+connect_token] for i in range(0,max_length,connect_token)]
                sublist_chunks = [sublist_chunks[i::chunk_num] for i in range(chunk_num)]
                sublist_chunks = torch.tensor(sublist_chunks).reshape(chunk_num, args.chunk_size)
                gap_list.append(sublist_chunks.tolist())
            batch["input_ids"] = torch.tensor(gap_list)
            connect_input_list=[original_input_order[i: i + connect_token] for i in range(0, max_length, connect_token)]
            original_input_list = [connect_input_list[i::chunk_num] for i in range(chunk_num)]
            input_index=sum(sum(original_input_list, []), [])
            inverse_indices = {index: i for i, index in enumerate(input_index)}
            inverse_indices_list = [inverse_indices[i] for i in range(max_length)]
            batch["inverse_input_list"]=inverse_indices_list
        label_ids = torch.zeros((len(features), num_labels))
        for i, f in enumerate(features):
            for label in f["label_ids"]:
                label_ids[i, label] = 1
        batch["label_ids"] = label_ids
        return batch

    def data_collator(features):
        batch = dict()
        max_length = max([len(f["input_ids"]) for f in features])
        # max_length=args.max_seq_length
        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size
        list_input_ids = [
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]
        original_input_order = list(range(max_length))
        if args.gap_type==0:
            batch["input_ids"] = torch.tensor(list_input_ids).contiguous().view((len(features), -1, args.chunk_size))
            batch["inverse_input_list"]=original_input_order
        elif args.gap_type==1:
            chunk_num=max_length // args.chunk_size
            gap_list = []
            for sublist in list_input_ids:
                sublist_chunks = [sublist[i::chunk_num] for i in range(chunk_num)]
                gap_list.append(sublist_chunks)
            batch["input_ids"]=torch.tensor(gap_list)
            original_input_list = [original_input_order[i::chunk_num] for i in range(chunk_num)]
            original_input_list=sum(original_input_list, [])
            inverse_indices = {index: i for i, index in enumerate(original_input_list)}
            inverse_indices_list = [inverse_indices[i] for i in range(max_length)]
            batch["inverse_input_list"]=inverse_indices_list
        elif args.gap_type == 2:
            chunk_num=max_length // args.chunk_size
            connect_token=args.connect_token
            connect_gap=args.chunk_size//connect_token
            gap_list = []
            for sublist in list_input_ids:
                sublist_chunks = [sublist[i:i+connect_token] for i in range(0,max_length,connect_token)]
                sublist_chunks = [sublist_chunks[i::chunk_num] for i in range(chunk_num)]
                sublist_chunks = torch.tensor(sublist_chunks).reshape(chunk_num, args.chunk_size)
                gap_list.append(sublist_chunks.tolist())
            batch["input_ids"] = torch.tensor(gap_list)
            connect_input_list=[original_input_order[i: i + connect_token] for i in range(0, max_length, connect_token)]
            original_input_list = [connect_input_list[i::chunk_num] for i in range(chunk_num)]
            input_index=sum(sum(original_input_list, []), [])
            inverse_indices = {index: i for i, index in enumerate(input_index)}
            inverse_indices_list = [inverse_indices[i] for i in range(max_length)]
            batch["inverse_input_list"]=inverse_indices_list
        label_ids = torch.zeros((len(features), num_labels))
        for i, f in enumerate(features):
            for label in f["label_ids"]:
                label_ids[i, label] = 1
        batch["label_ids"] = label_ids
        return batch

    processed_datasets = raw_datasets.map(getitem, batched=True,
                                          remove_columns=remove_columns)
    train_dataloader = DataLoader(processed_datasets["train"], shuffle=True, collate_fn=data_collator_train,
                                  batch_size=args.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=data_collator, batch_size=args.batch_size,
                                 pin_memory=True)
    test_dataloader = DataLoader(processed_datasets["test"], collate_fn=data_collator, batch_size=args.batch_size,
                                 pin_memory=True)
    if args.optimiser.lower() == "adamw":
        if args.use_different_lr:
            ignored_params = list(map(id, model.roberta.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer_grouped_parameters = [
                {
                    "params": model.roberta.parameters(),
                    "lr": args.plm_lr,
                },
                {
                    "params": base_params,
                    "lr": args.lr,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        else:
            betas = (0.9, 0.999)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, betas=betas, weight_decay=args.weight_decay)
            print("optimizer", optimizer)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.n_epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    T_epochs = args.num_train_epochs
    if args.use_lr_scheduler:
        itersPerEpoch = num_update_steps_per_epoch
        print("itersPerEpoch", itersPerEpoch)
        epoch = T_epochs
        warmupEpochs = args.warmup
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmupEpochs * itersPerEpoch,
                                                       num_training_steps=epoch * itersPerEpoch)
    criterions = nn.BCEWithLogitsLoss()
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    batch_id = 0
    metrics_max = None
    metrics_max_val = -1
    epoch_max = 0
    for epoch in tqdm(range(args.n_epoch)):
        print(" ")
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        model.train()
        optimizer.zero_grad()
        losses = []
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch_id += 1
            outputs,att = model(**batch)
            loss = criterions(outputs, batch['label_ids'])
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            losses.append(loss.item())
            epoch_loss += loss.item()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix(loss=epoch_loss / batch_id)
        model.eval()
        all_preds = []
        all_preds_raw = []
        all_labels = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs,att = model(**batch)
            preds_raw = outputs.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["label_ids"].cpu().numpy()))
        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
        print("lr{}:".format(optimizer.param_groups[0]["lr"]),
              "lr{}:".format(optimizer.param_groups[1]["lr"]), "loss: ", np.mean(losses).item())
        print(
            f" F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
        print(" ")
        if metrics_max_val < metrics['f1_micro']:
            epoch_max = epoch + 1
            metrics_max_val = metrics['f1_micro']
            if args.best_model_path is not None:
                os.makedirs(args.best_model_path, exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.best_model_path, save_function=accelerator.save)
            checkpoint = {'epoch': epoch + 1,
                          'metrics': metrics,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict()}
            torch.save(checkpoint, args.best_model_path + "check.pth")
    print("best epoch:", epoch_max)

if __name__ == "__main__":
    main()

