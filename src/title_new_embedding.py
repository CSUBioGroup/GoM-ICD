from src.args_parse import *
from src.process import *
import numpy as np
import pandas as pd


from transformers import AutoTokenizer,RobertaModel
import torch.nn as nn

def reformat_icd(code: str, version: int, is_diag: bool) -> str:
    """format icd code depending on version"""
    if version == 9:
        return reformat_icd9(code, is_diag)
    elif version == 10:
        return reformat_icd10(code, is_diag)
    else:
        raise ValueError("version must be 9 or 10")


def reformat_icd10(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if not is_diag:
        return code
    return code[:3] + "." + code[3:]


def reformat_icd9(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if is_diag:
        if code.startswith("E"):
            if len(code) > 4:
                return code[:4] + "." + code[4:]
        else:
            if len(code) > 3:
                return code[:3] + "." + code[3:]
    else:
        if len(code) > 2:
            return code[:2] + "." + code[2:]
    return code


import os

def init_title_mimiciv(label2index):

    ICD_dic=set()
    mimicPath = "path"
    dIcdDiagnoses = pd.read_csv(os.path.join(mimicPath, 'D_ICD_DIAGNOSES.csv'), dtype=str)
    for dic, code in dIcdDiagnoses['icd_code'].items():
        if dIcdDiagnoses.loc[dic, 'icd_version']=="9":
            dIcdDiagnoses.loc[dic, 'icd_code'] = reformat_icd(code=str(code), version=9,is_diag=True)
        elif dIcdDiagnoses.loc[dic, 'icd_version']=="10":
            dIcdDiagnoses.loc[dic, 'icd_code'] = reformat_icd(code=str(code), version=10, is_diag=True)
        ICD_dic.add(dIcdDiagnoses.loc[dic, 'icd_code'])
    dIcdDiagnoses = dIcdDiagnoses.set_index('icd_code')
    dicdProcedures = pd.read_csv(os.path.join(mimicPath, 'D_ICD_PROCEDURES.csv'), dtype=str)
    dicdProcedures['icd_code'] = dicdProcedures['icd_code'].astype(str)
    for dic, code in dicdProcedures['icd_code'].items():
        if dicdProcedures.loc[dic, 'icd_version']=="9":
            dicdProcedures.loc[dic, 'icd_code'] = reformat_icd(code=str(code), version=9,is_diag= False)
        elif dicdProcedures.loc[dic, 'icd_version'] == "10":
            dicdProcedures.loc[dic, 'icd_code'] = reformat_icd(code=str(code), version=10, is_diag=False)
        ICD_dic.add(dicdProcedures.loc[dic, 'icd_code'])
    dicdProcedures = dicdProcedures.set_index('icd_code')
    icdTitles = pd.concat([dIcdDiagnoses, dicdProcedures])
    title_icd=dict()
    titles = []

    for icd in label2index:
        try:
            desc = (icd+" : "+ icdTitles.loc[icd]['long_title']).lower()
        except:
            desc = (icd+ " <:> ").lower()
        desc=desc+" <EOS> "
        titles.append(desc)
        title_icd[icd]=desc
    return title_icd


def get_emb_word2vec(title_icd,label2index,model_name_or_path=None,method="skipgram",title_fea_size=128,
                     tokenizer=None,type_icd="icd9"):
    id2descVec = np.zeros((len(label2index), title_fea_size), dtype='float32')

    title_list=[]
    title_count = Counter()
    x=0
    for icd,desc in title_icd.items():
        icd_title_list =desc.split()
        title_list.append(icd_title_list)
        title_count.update(icd_title_list)
    if model_name_or_path is None:
        if method=="skipgram":
            model = Word2Vec(title_list, min_count=0, window=5, vector_size=title_fea_size, workers=8, sg=1, epochs=10)
        elif method=="cwob":
            model = Word2Vec(title_list, min_count=0, window=5, vector_size=title_fea_size, workers=8, sg=0, epochs=10)
    else:
        with open(model_name_or_path, 'rb') as f:
            titleEmbedding = pickle.load(f)
        title_emb_layer=nn.Embedding.from_pretrained(torch.tensor(titleEmbedding, dtype=torch.float32), freeze=False)
        tokenizedTitle = torch.LongTensor(tokenizer)
        title_vec = title_emb_layer(tokenizedTitle).detach().data.cpu().numpy().mean(axis=1)
        for i in range(len(title_vec)):
            id2descVec[i]=title_vec[i]
        return id2descVec
    title = sorted(title_count.keys())
    title_to_id = {v: i for i, v in enumerate(title)}
    id_to_title={i: v for i, v in enumerate(title)}
    len_title=(len(title))
    tword2vec = np.zeros((len_title, title_fea_size), dtype=np.float32)
    for i in range(len_title):
        tword2vec[i] = model.wv[id_to_title[i]]

    if type_icd=="icd9":
        path='path.pkl'%(method,title_fea_size)
    elif type_icd=="icd10":
        path = 'path.pkl' % (method, title_fea_size)
    with open(path, 'wb') as f:
        pickle.dump(tword2vec, f, protocol=4)

    title_emb_layer=nn.Embedding.from_pretrained(torch.tensor(tword2vec, dtype=torch.float32), freeze=False)
    for icd in label2index:
        tokenized_title = np.array([title_to_id[token] for token in title_icd[icd].split()],dtype='int32')
        tokenizedTitle = torch.LongTensor(tokenized_title)
        title_vec = title_emb_layer(tokenizedTitle).detach().data.cpu().numpy().mean(axis=0)
        id2descVec[label2index[icd]] = title_vec

    return id2descVec


def get_emb_roberta(title_icd,label2index,model_name_or_path):
    id2descVec = np.zeros((len(label2index), 768), dtype='float32')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = RobertaModel.from_pretrained(model_name_or_path).eval().to('cuda:0')
    for icd,desc in title_icd.items():
        titleLen = len(desc)
        inputs = tokenizer(desc,padding=False,max_length=titleLen,truncation=True,)
        tokens = tokenizer.tokenize(desc)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids]).to('cuda:0')
        tmp=model(input_ids)
        tmp=tmp[0].detach().data.cpu().numpy().mean(axis=1)
        id2descVec[label2index[icd]] = tmp
    return id2descVec


def train_emb_title(label2index,title_emb_mode,title_fea_size=None, model_name_or_path=None,type_icd="icd9"):
    title_icd = init_title_mimiciv(label2index)
    if title_emb_mode == 'skipgram' or title_emb_mode == 'cbow':
        id2descVec=get_emb_word2vec(title_icd, label2index, model_name_or_path=model_name_or_path,
                                    method=title_emb_mode, title_fea_size=title_fea_size,type_icd=type_icd)
        data = {'id2descVec': id2descVec}
        data['label_to_id'] = label2index
        if type_icd=="icd9":
            with open('path.pkl', 'wb') as file:
                pickle.dump(data, file)
        elif type_icd=="icd10":
            with open('path.pkl', 'wb') as file:
                pickle.dump(data, file)
    elif title_emb_mode=='roberta-PM':
        model_name_or_path="path"
        id2descVec=get_emb_roberta(title_icd, label2index, model_name_or_path)
        data = {'id2descVec':id2descVec}
        data['label_to_id']=label2index

        if type_icd == "icd9":
            with open('path.pkl', 'wb') as file:
                pickle.dump(data, file)
        elif type_icd == "icd10":
            with open('path.pkl', 'wb') as file:
                pickle.dump(data, file)

from src.vocab_new import *
def emb_new_data(type_icd="icd9",title_emb_mode = "skipgram"):
    args = create_args_parser()
    if type_icd=="icd9":
        args.data_dir = "../data/mimicdata/mimiciv_icd9/"
    elif type_icd == "icd10":
        args.data_dir = "../data/mimicdata/mimiciv_icd10/"
    if title_emb_mode == "skipgram":
        raw_datasets = load_data(args.data_dir)
        vocab = Vocab_new(args, raw_datasets)
        label_to_id = vocab.label_to_id
        train_emb_title(label_to_id, title_emb_mode="skipgram", title_fea_size=1024,type_icd=type_icd)
    elif title_emb_mode == "roberta-PM":
        raw_datasets = load_data(args.data_dir)
        vocab = Vocab_new(args, raw_datasets)
        label_to_id = vocab.label_to_id
        train_emb_title(label_to_id, title_emb_mode='roberta-PM')

if __name__ == "__main__":
    emb_new_data(type_icd="icd9",title_emb_mode = "skipgram")


