import numpy as np
from collections import Counter
from gensim.models.word2vec import Word2Vec
import torch
class Vocab(object):
    def __init__(self,args,raw_datasets):
        self.token_len=0
        self.label_len=0

        self.word2index = dict()
        self.index2word = dict()
        self.label_dict=dict()
        self.raw_data=raw_datasets
        self.columns_name = raw_datasets["train"].column_names
        self.label_count,self.label_to_id=self.get_label()
        self.label_num=len(self.label_to_id)
        self.build_vocab()
        self.word_num = len(self.word2index)
        self.word_embeddings=self.build_embedding(word_embedding_file=args.word_embedding_file)


    def get_label(self):
        # lables=set()
        # lable_list=[]
        label_count=Counter()
        if "Full_Labels" in self.columns_name:
            label_columns="Full_Labels"
        elif "target" in self.columns_name:
            label_columns = "target"
        for data in self.raw_data:
            self.label_dict[data]=Counter()
            for row_da_item in self.raw_data[data]:
                if "Full_Labels"==label_columns:
                    lable_list=row_da_item['Full_Labels'].split('|')
                elif "target"==label_columns:
                    lable_list = row_da_item['target'].split(",")
                # print(lable_list)
                label_count.update(lable_list)
                self.label_dict[data].update(lable_list)

                self.label_len+=len(lable_list)
        # print(lable_list)
        # print(label_count)
        # print(len(label_count))
        # print(type(label_count.keys()))
        # print(len(label_count.keys()))
        labels = sorted(label_count.keys())
        label_to_id = {v: i for i, v in enumerate(labels)}
        return label_count,label_to_id

    def build_vocab(self):
        word_count = Counter()
        if "Text" in self.columns_name:
            text_columns="Text"
        elif "text" in self.columns_name:
            text_columns = "text"
        for data in self.raw_data:
            for row_da_item in self.raw_data[data]:
                #取数据集中的文本
                text_list=row_da_item[text_columns].split()
                word_count.update(text_list)

                self.token_len+=len(text_list)
                # print(text_list)
                # print("类型",type(text_list))
                # print("长度",len(text_list))

        # print(counter)72363-50
        print(len(word_count.keys()))
        words = sorted(word_count.keys())
        self.word2index = {word: idx+2 for idx, word in enumerate(words)}
        self.word2index['_PAD']=0
        self.word2index['_UNK']=1
        self.index2word = {idx+2: word for idx, word in enumerate(words)}
        self.index2word[0]='_PAD'
        self.index2word[1]='_UNK'

    def build_embedding(self,word_embedding_file):
        model = Word2Vec.load(word_embedding_file)
        embedding_size = model.wv["and"].size
        # print("model.wv["and"]的type",model.wv["and"])
        # print("model.wv["and"]的type",type(model.wv["and"]))
        # print("embedding_size“:",embedding_size)
        unknown_vec = np.random.uniform(-0.25, 0.25, embedding_size)
        #un_know 单词设置为随机，并初始化所有embedding
        embeddings = [unknown_vec] * (self.word_num)
        #embedding_len“: 72365
        # print("embedding_len“:",len(embeddings))
        #pad设置为0
        embeddings[0] = np.zeros(embedding_size)
        unknown_num=0
        for word in self.word2index:
            try:
                embeddings[self.word2index[word]] = model.wv[word]
            except:
                # print(word)
                unknown_num+=1
                # self.word2index[word] = self.word2index[self.UNK_TOKEN]
                pass
        embeddings = torch.FloatTensor(np.array(embeddings,dtype=np.float32))
        # print("未知词个数：",unknown_num)

        return embeddings
    def index_of_word(self,word):
        try:
            return self.word2index[word]
        except:
            return self.word2index['_UNK']



