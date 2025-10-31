from src.args_parse import *
import numpy as np
import pandas as pd
import os
from src.process import *
from transformers import AutoTokenizer,RobertaModel
import torch.nn as nn
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def init_title_mimiciii_clean(label2index):
    # tword2id, id2tword = {"<EOS>": 0}, ["<EOS>"]
    ICD_dic=set()
    mimicPath = "data/mimicdata/icd_title/"
    dIcdDiagnoses = pd.read_csv(os.path.join(mimicPath, 'D_ICD_DIAGNOSES.csv'), dtype=str)
    dIcdDiagnoses['ICD9_CODE'] = dIcdDiagnoses['ICD9_CODE'].astype(str)
    for dic, code in dIcdDiagnoses['ICD9_CODE'].items():
        dIcdDiagnoses.loc[dic, 'ICD9_CODE'] = reformat_icd(code=str(code), version=9,is_diag=True)
        ICD_dic.add(dIcdDiagnoses.loc[dic, 'ICD9_CODE'])
    dIcdDiagnoses = dIcdDiagnoses.set_index('ICD9_CODE')
    dicdProcedures = pd.read_csv(os.path.join(mimicPath, 'D_ICD_PROCEDURES.csv'), dtype=str)
    dicdProcedures['ICD9_CODE'] = dicdProcedures['ICD9_CODE'].astype(str)
    for dic, code in dicdProcedures['ICD9_CODE'].items():
        dicdProcedures.loc[dic, 'ICD9_CODE'] = reformat_icd(code=str(code), version=9,is_diag= False)
        ICD_dic.add(dicdProcedures.loc[dic, 'ICD9_CODE'])
    dicdProcedures = dicdProcedures.set_index('ICD9_CODE')
    icdTitles = pd.concat([dIcdDiagnoses, dicdProcedures])
    title_icd=dict()
    titles = []


    for icd in label2index:
        try:
            # 标签长描述和短描述拼接
            desc = (icd+" : "+icdTitles.loc[icd]['SHORT_TITLE'] + ' <:> ' + icdTitles.loc[icd]['LONG_TITLE']).lower()
        except:
            flag=False
            for i in range(10):
                if '.' not in icd:
                    icd_fake = icd + '.'+str(i)
                else:
                    icd_fake=icd+str(i)
                if icd_fake in ICD_dic:
                    flag=True
                    desc = (icd+" : "+icdTitles.loc[icd_fake]['SHORT_TITLE'] + ' <:> ' + icdTitles.loc[icd_fake]['LONG_TITLE']).lower()
                    break
            if not flag:
                desc = (icd+ " <:> ").lower()

        desc=desc+" <EOS> "
        titles.append(desc)
        title_icd[icd]=desc
    return title_icd

def get_emb_word2vec(title_icd,label2index,model_name_or_path=None,method="skipgram",title_fea_size=128,tokenizer=None):
    id2descVec = np.zeros((len(label2index), title_fea_size), dtype='float32')
    title_list=[]
    title_count = Counter()
    x=0
    for icd,desc in title_icd.items():
        icd_title_list =desc.split()
        if x ==0:
            print(icd_title_list)
            x+=1
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

    #path="../data/mimicdata/mimiciii_clean/"
    path='path.pkl'
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
    model = RobertaModel.from_pretrained(model_name_or_path).eval().to(device)
    for icd,desc in title_icd.items():
        titleLen = len(desc)
        inputs = tokenizer(desc,padding=False,max_length=titleLen,truncation=True,)
        tokens = tokenizer.tokenize(desc)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([ids]).to(device)
        tmp=model(input_ids)
        tmp=tmp[0].detach().data.cpu().numpy().mean(axis=1)
        id2descVec[label2index[icd]] = tmp
    return id2descVec



def train_emb_title(label2index,title_emb_mode,title_fea_size=None, model_name_or_path=None,
                    is_50=False,label_to_id_50=None):

    title_icd = init_title_mimiciii_clean(label2index)
    if title_emb_mode == 'skipgram' or title_emb_mode == 'cbow':
        id2descVec=get_emb_word2vec(title_icd, label2index, model_name_or_path=model_name_or_path,
                                    method=title_emb_mode, title_fea_size=title_fea_size)
        data = {'id2descVec': id2descVec}
        data['label_to_id'] = label2index
        with open('path.pkl' , 'wb') as file:
            pickle.dump(data, file)
    elif title_emb_mode=='roberta-PM':
        model_name_or_path="models/RoBERTa-base-PM-M3-Voc-distill-align-hf/"
        id2descVec=get_emb_roberta(title_icd, label2index, model_name_or_path)
        data = {'id2descVec':id2descVec}
        data['label_to_id']=label2index
#

        with open('path.pkl', 'wb') as file:
            pickle.dump(data, file)

from src.vocab_all import *
def emb_new_data(title_emb_mode):
    if title_emb_mode == "skipgram":
        args = create_args_parser()
        args.data_dir = "path"
        raw_datasets = load_data(args.data_dir)
        vocab = Vocab(args, raw_datasets)
        label_to_id = vocab.label_to_id
        train_emb_title(label_to_id, title_emb_mode="skipgram", title_fea_size=1024)
    elif title_emb_mode == "roberta-PM":
        args = create_args_parser()
        args.data_dir = "path"
        raw_datasets = load_data(args.data_dir)
        vocab = Vocab(args, raw_datasets)
        label_to_id = vocab.label_to_id
        train_emb_title(label_to_id, title_emb_mode='roberta-PM')

if __name__ == "__main__":

    emb_new_data(title_emb_mode = "skipgram")


