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
from src.function_lable import *

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

    if "Text" in remove_columns:
        text_name = "Text"
    elif "text" in remove_columns:
        text_name = "text"
    if "Full_Labels" in remove_columns:
        label_columns = "Full_Labels"
    elif "target" in remove_columns:
        label_columns = "target"

    def getitem(examples):
        label_list = []
        texts = ((examples[text_name],))

        result = tokenizer(*texts, padding=False, max_length=args.max_seq_length, truncation=True,
                           add_special_tokens=True)
        batch_encoding = {"input_ids": result["input_ids"]}
        # batch_decoding = tokenizer.batch_decode(batch_encoding["input_ids"][1])
        if "Full_Labels" == label_columns:
            for labels in examples["Full_Labels"]:
                label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split('|')])
        elif "target" == label_columns:
            for labels in examples["target"]:
                label_list.append([vocab.label_to_id[label.strip()] for label in labels.strip().split(',')])
        result["label_ids"] = label_list
        return result
    def data_collator(features,gap_type):
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
        if gap_type==0:
            batch["input_ids"] = torch.tensor(list_input_ids).contiguous().view((len(features), -1, args.chunk_size))
            batch["inverse_input_list"]=original_input_order

        elif gap_type==1:
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
        elif gap_type == 2:
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

    # best_model_path1 =args.add_path_gap0
    if args.add_path_gap0 is not None:
        check = torch.load(args.add_path_gap0 + "check.pth", map_location=torch.device('cuda:2'))
        print()
        print(check['epoch'])
        print(check['metrics'])
        print()
        model = Roberta_model.from_pretrained(
            args.add_path_gap0,
            config=config,
            args=args,
            vocab=vocab
        )
        train_dataloader = DataLoader(processed_datasets["train"], collate_fn=lambda batch: data_collator(batch,0),
                                      batch_size=args.batch_size, pin_memory=True)
        eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=lambda batch: data_collator(batch,0),
                                     batch_size=args.batch_size,pin_memory=True)
        test_dataloader = DataLoader(processed_datasets["test"], collate_fn=lambda batch: data_collator(batch,0),
                                     batch_size=args.batch_size,pin_memory=True)
        train_dataloader = accelerator.prepare(train_dataloader)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        test_dataloader = accelerator.prepare(test_dataloader)
        model = accelerator.prepare(model)

        label_vec1 = get_test_gap(model,test_dataloader)

        metrics_max_2, best_thre_2 = ans_test(test_dataloader, model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.add_path_gap1 is not None:
        check = torch.load(args.add_path_gap1 + "check.pth", map_location=torch.device('cuda:2'))
        print()
        print(check['epoch'])
        print(check['metrics'])
        print()
        model = Roberta_model.from_pretrained(
            args.add_path_gap1,
            config=config,
            args=args,
            vocab=vocab
        )
        train_dataloader = DataLoader(processed_datasets["train"], collate_fn=lambda batch: data_collator(batch,1),
                                      batch_size=args.batch_size, pin_memory=True)
        eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=lambda batch: data_collator(batch,1),
                                     batch_size=args.batch_size,pin_memory=True)
        test_dataloader = DataLoader(processed_datasets["test"], collate_fn=lambda batch: data_collator(batch,1),
                                     batch_size=args.batch_size,pin_memory=True)
        train_dataloader = accelerator.prepare(train_dataloader)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        test_dataloader = accelerator.prepare(test_dataloader)
        model = accelerator.prepare(model)
        label_vec2 = get_test_gap(model, test_dataloader)
        metrics_max_2, best_thre_2 = ans_test(test_dataloader, model)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # ================================================================================
    if args.add_path_gap2 is not None:
        check = torch.load(args.add_path_gap2 + "check.pth", map_location=torch.device('cuda:2'))
        print()
        print(check['epoch'])
        print(check['metrics'])
        print()
        model = Roberta_model.from_pretrained(
            args.add_path_gap2,
            config=config,
            args=args,
            vocab=vocab
        )
        train_dataloader = DataLoader(processed_datasets["train"], collate_fn=lambda batch: data_collator(batch,2),
                                      batch_size=args.batch_size, pin_memory=True)
        eval_dataloader = DataLoader(processed_datasets["valid"], collate_fn=lambda batch: data_collator(batch,2),
                                     batch_size=args.batch_size,pin_memory=True)
        test_dataloader = DataLoader(processed_datasets["test"], collate_fn=lambda batch: data_collator(batch,2),
                                     batch_size=args.batch_size,pin_memory=True)
        train_dataloader = accelerator.prepare(train_dataloader)
        eval_dataloader = accelerator.prepare(eval_dataloader)
        test_dataloader = accelerator.prepare(test_dataloader)
        model = accelerator.prepare(model)
        metrics_max_2, best_thre_2 = ans_test(test_dataloader, model)
        label_vec3 = get_test_gap(model, test_dataloader)
        # label_vec3 = get_label_vec_out(model, train_dataloader, eval_dataloader, test_dataloader)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 这里直接给出我的一个结果
    # if args.add_path_gap2 is not None and args.add_path_gap1 is not None and args.add_path_gap0 is not None:
    #     a=0.6
    #     b=0.1
    #     c=0.3
    #     preds_raw=a*label_vec1['test_input_tensor']+b*label_vec2['test_input_tensor']+c*label_vec3['test_input_tensor']
    #     all_preds_raw =preds_raw
    #     all_labels=label_vec1['test_target_tensor']
    #     list_threshold = [0.3, 0.35, 0.4, 0.45, 0.5]
    #     for t in list_threshold:
    #         all_preds = (all_preds_raw > t).astype(int)
    #         metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5, 8, 15])
    #         print(f"F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
    #         print(f"prec_micro:{metrics['prec_micro']:.4f},rec_micro:{metrics['rec_micro']:.4f}")
    #         print("   ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    #         print()
    #     print("  ")
    #     preds_raw=a*label_vec1['outputs_np']+b*label_vec2['outputs_np']+c*label_vec3['outputs_np']
    #     all_preds_raw = torch.tensor(preds_raw).sigmoid().numpy()
    #     all_labels=label_vec1['test_target_tensor']
    #     list_threshold = [0.3, 0.35, 0.4, 0.45, 0.5]
    #     for t in list_threshold:
    #         all_preds = (all_preds_raw > t).astype(int)
    #         metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5, 8, 15])
    #         print(f"F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
    #         print(f"prec_micro:{metrics['prec_micro']:.4f},rec_micro:{metrics['rec_micro']:.4f}")
    #         print("   ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
    #         print()

if __name__ == "__main__":
    main()

