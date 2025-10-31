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
        label_count=Counter()
        for data in self.raw_data:
            self.label_dict[data]=Counter()
            for row_da_item in self.raw_data[data]:
                lable_list = row_da_item['target'].split(",")  #FUll_label
                label_count.update(lable_list)
                self.label_dict[data].update(lable_list)
                self.label_len+=len(lable_list)
        labels = sorted(label_count.keys())
        label_to_id = {v: i for i, v in enumerate(labels)}
        return label_count,label_to_id

    def build_vocab(self):
        word_count = Counter()
        for data in self.raw_data:
            for row_da_item in self.raw_data[data]:
                text_list=row_da_item['text'].split()
                word_count.update(text_list)
                self.token_len+=len(text_list)

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
        unknown_vec = np.random.uniform(-0.25, 0.25, embedding_size)
        embeddings = [unknown_vec] * (self.word_num)
        embeddings[0] = np.zeros(embedding_size)
        unknown_num=0
        for word in self.word2index:
            try:
                embeddings[self.word2index[word]] = model.wv[word]
            except:
                pass
        embeddings = torch.FloatTensor(np.array(embeddings,dtype=np.float32))

        return embeddings
    def index_of_word(self,word):
        try:
            return self.word2index[word]
        except:
            return self.word2index['_UNK']



