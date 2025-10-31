import torch
from tqdm.autonotebook import tqdm
from torch.cuda.amp import autocast,GradScaler
from src.evaluation import all_metrics
import numpy as np
from torch.utils.data import Dataset
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler=GradScaler()
class TextDataset(Dataset):
    def __init__(self,text_data,max_seq_length,vocab,sort = False,):
        super(TextDataset, self).__init__()
        self.max_seq_length=max_seq_length
        self.vocab = vocab
        indexed_data = []
        self.n_instances = len(text_data)
        self.n_total_tokens = 0
        self.label_count = dict()
        self.labels = set()
        for features in tqdm(text_data, desc="Processing data"):
            label_id_list=[]
            label_list =[label for label in features["Full_Labels"].strip().split('|')]
            for label in label_list:
                label_id = self.vocab.label_to_id[label]
                if label_id not in self.label_count:
                    self.label_count[label_id] = 1
                else:
                    self.label_count[label_id] += 1
                    self.labels.add(label_id)
                    label_id_list.append(label_id)
            word_seq = []

            records = features["Text"].split("\n\n")
            should_break = False
            for i in range(len(records)):
                record = records[i]
                sentences = record.split("\n")

                for sentence in sentences:
                    sent_words = sentence.strip().split()
                    if len(sent_words) == 0:
                        continue
                    for word in sent_words:
                        word_idx = vocab.index_of_word(word)
                        word_seq.append(word_idx)
                        self.n_total_tokens += 1
                        if len(word_seq) >= self.max_seq_length > 0:
                            should_break = True
                            break
                    if should_break:
                        break

                if should_break:
                    break
            if len(word_seq) > 0:
                indexed_data.append((word_seq, label_id_list))

        if sort:
            self.indexed_data = sorted(indexed_data, key=lambda x: -len(x[0]))
        else:
            self.indexed_data =indexed_data
        self.labels = sorted(list(self.labels))

    def __getitem__(self, index):
        word_seq, label_list  = self.indexed_data[index]
        result=dict()
        result["input_ids"] = word_seq
        result["label_ids"] = label_list
        return result

    def __len__(self):
        return len(self.indexed_data)


def ans_test(test_dataloader,model):
    model.eval()
    all_preds = []
    all_preds_raw = []
    all_labels = []
    metrics_max=None
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            if model.name=="plm_model":
                outputs,att = model(**batch)
            elif model.name=="rnn_model":
                outputs = model(batch)
        preds_raw = outputs.sigmoid().cpu()
        preds = (preds_raw > 0.5).int()
        all_preds_raw.extend(list(preds_raw))
        all_preds.extend(list(preds))
        all_labels.extend(list(batch["label_ids"].cpu().numpy()))

    all_preds_raw = np.stack(all_preds_raw)
    all_preds = np.stack(all_preds)
    all_labels = np.stack(all_labels)
    metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
    list_threshold=[0.3, 0.35, 0.4, 0.45, 0.5]
    best_thre=0
    for t in list_threshold:
        all_preds = (all_preds_raw > t).astype(int)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5, 8, 15])
        if metrics_max is None:
            metrics_max=metrics
            best_thre=t
        else:
            if metrics_max['f1_micro']<metrics['f1_micro']:
                metrics_max=metrics
                best_thre=t
    return metrics_max,best_thre


def ans_test_all(test_dataloader,model):
    model.eval()
    ma, opt_thr = -1, -1
    for thr in np.linspace(0.3, 0.6, 20):
        all_preds = []
        all_preds_raw = []
        all_labels = []
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                if model.name == "plm_model":
                    outputs,att = model(**batch)
                elif model.name == "rnn_model":
                    outputs = model(batch)
            preds_raw = outputs.sigmoid().cpu()
            preds = (preds_raw > thr).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["label_ids"].cpu().numpy()))
        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
        if metrics['f1_micro'] > ma:
            ma = metrics['f1_micro']
            opt_thr = thr
    all_preds = (all_preds_raw > opt_thr).astype(int)
    metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5, 8, 15])
    return metrics,opt_thr


