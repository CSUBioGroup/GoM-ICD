import numpy as np
import torch
import datetime
from src.evaluation import *
import gc
import pickle

def get_label_vec(model,train_dataloader,eval_dataloader, test_dataloader,short_lists,j):
    label_vec = dict()
    # =====================================================================
    model.eval()
    all_labels = []
    att_list = []
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            outputs, att = model(**batch)
        all_labels.extend(batch["label_ids"][:, short_lists[j]].cpu())
        att_list.extend(att[:, short_lists[j], :].cpu())
    label_vec['train_input_tensor'] = torch.stack(att_list)
    label_vec['train_target_tensor'] = torch.stack(all_labels)
    del att_list, all_labels
    del outputs,att
    gc.collect()
    all_labels = []
    att_list = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs, att = model(**batch)
        all_labels.extend(batch["label_ids"][:, short_lists[j]].cpu())
        att_list.extend(att[:, short_lists[j], :].cpu())
    label_vec['val_input_tensor'] = torch.stack(att_list)
    label_vec['val_target_tensor'] = torch.stack(all_labels)
    del att_list, all_labels
    del outputs,att
    gc.collect()
    all_labels = []
    att_list = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs, att = model(**batch)
        all_labels.extend(batch["label_ids"][:, short_lists[j]].cpu())
        att_list.extend(att[:, short_lists[j], :].cpu())
    label_vec['test_input_tensor'] = torch.stack(att_list)
    label_vec['test_target_tensor'] = torch.stack(all_labels)
    del att_list, all_labels
    del outputs,att
    gc.collect()
    return label_vec

def get_test_semble(model_semble,test_semble_dataloader):
    model_semble.eval()
    all_preds = []
    all_preds_raw = []
    all_labels = []
    for step, batch in enumerate(test_semble_dataloader):
        with torch.no_grad():
            outputs = model_semble(batch[0])
        preds_raw = outputs.sigmoid().cpu()
        preds = (preds_raw > 0.5).int()
        all_preds_raw.extend(list(preds_raw))
        all_preds.extend(list(preds))
        all_labels.extend(list(batch[1].cpu().numpy()))
    all_preds_raw = np.stack(all_preds_raw)
    all_preds = np.stack(all_preds)
    all_labels = np.stack(all_labels)
    return all_preds_raw,all_preds,all_labels

def get_label_vec_out(model,train_dataloader,eval_dataloader, test_dataloader):
    label_vec = dict()
    model.eval()
    all_preds = []
    all_preds_raw = []
    all_labels = []
    outputs_list=[]
    for step, batch in enumerate(train_dataloader):
        with torch.no_grad():
            outputs, att = model(**batch)
            outputs_list.extend(list(outputs.cpu()))
            preds_raw = outputs.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["label_ids"].cpu().numpy()))
    all_preds_raw = np.stack(all_preds_raw)
    all_preds = np.stack(all_preds)
    all_labels = np.stack(all_labels)
    outputs_np=np.stack(outputs_list)
    label_vec['train_outputs_np'] = outputs_np
    label_vec['train_target_tensor'] = all_labels
    metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)

    all_preds = []
    all_preds_raw = []
    all_labels = []
    outputs_list=[]

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs, att = model(**batch)
            outputs_list.extend(list(outputs.cpu()))
            preds_raw = outputs.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["label_ids"].cpu().numpy()))
    all_preds_raw = np.stack(all_preds_raw)
    all_preds = np.stack(all_preds)
    all_labels = np.stack(all_labels)
    outputs_np=np.stack(outputs_list)
    label_vec['val_outputs_np'] = outputs_np
    label_vec['val_target_tensor'] = all_labels
    metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)

    all_preds = []
    all_preds_raw = []
    all_labels = []
    outputs_list=[]
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs, att = model(**batch)
            outputs_list.extend(list(outputs.cpu()))
            preds_raw = outputs.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["label_ids"].cpu().numpy()))
    all_preds_raw = np.stack(all_preds_raw)
    all_preds = np.stack(all_preds)
    all_labels = np.stack(all_labels)
    outputs_np=np.stack(outputs_list)

    label_vec['test_target_tensor'] = all_labels
    label_vec['test_outputs_np'] = outputs_np
    metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)

    return label_vec

def get_test_gap(model,test_dataloader):
    model.eval()
    all_preds = []
    all_preds_raw = []
    all_labels = []
    outputs_list=[]
    label_vec=dict()
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs, att = model(**batch)
            outputs_list.extend(list(outputs.cpu()))
            preds_raw = outputs.sigmoid().cpu()
            preds = (preds_raw > 0.5).int()
            all_preds_raw.extend(list(preds_raw))
            all_preds.extend(list(preds))
            all_labels.extend(list(batch["label_ids"].cpu().numpy()))
    all_preds_raw = np.stack(all_preds_raw)
    all_preds = np.stack(all_preds)
    all_labels = np.stack(all_labels)
    outputs_np=np.stack(outputs_list)
    label_vec['test_input_tensor'] = all_preds_raw
    label_vec['test_target_tensor'] = all_labels
    label_vec['outputs_np'] = outputs_np
    print(datetime.datetime.now().strftime('%H:%M:%S'))
    metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
    print(
        f" F1_micro: {metrics['f1_micro']:.4f}, F1_macro: {metrics['f1_macro']:.4f}, auc_micro: {metrics['auc_micro']:.4f}, auc_macro: {metrics['auc_macro']:.4f}, prec_at_8: {metrics['prec_at_8']:.4f}")
    print(" ")
    return label_vec

def stor_label_vec(label_vec1, label_vec2,label_vec3,path):
    label_vec=dict()
    label_vec['train_outputs_np']=np.stack((label_vec1['train_outputs_np'],label_vec2['train_outputs_np'],label_vec3['train_outputs_np']), axis=2)
    label_vec['val_outputs_np']=np.stack((label_vec1['val_outputs_np'],label_vec2['val_outputs_np'],label_vec3['val_outputs_np']), axis=2)
    label_vec['test_outputs_np']=np.stack((label_vec1['test_outputs_np'],label_vec2['test_outputs_np'],label_vec3['test_outputs_np']), axis=2)
    label_vec['train_target_tensor']=label_vec1['train_target_tensor']
    label_vec['val_target_tensor']=label_vec1['val_target_tensor']
    label_vec['test_target_tensor']=label_vec1['test_target_tensor']
    with open(path, 'wb') as f:
        pickle.dump(label_vec, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_label_vec(path):
    with open(path, 'rb') as f:
        loaded_string = pickle.load(f)
    return loaded_string
