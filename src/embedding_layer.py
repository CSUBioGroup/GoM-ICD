import torch.nn as nn
import torch
import copy


class EmbeddingLayer(nn.Module):
    def __init__(self,embedding_mode,pretrained_word_embeddings,vocab_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding_size = pretrained_word_embeddings.shape[-1]
        requires_grad = False if embedding_mode == "static" else True
        self.embeddings = nn.Embedding(vocab_size, self.embedding_size)
        self.embeddings.weight = nn.Parameter(copy.deepcopy(pretrained_word_embeddings), requires_grad=requires_grad)
    def forward(self,batch_data):
        embeds = self.embeddings(batch_data)  # [batch_size x max_seq_size x embedding_size]
        return embeds

    
class Title_Emb_Layer(nn.Module):
    def __init__(self,embedding,freeze=False):
        super(Title_Emb_Layer, self).__init__()
        self.title_emb = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float32), freeze=freeze)
    def forward(self,title):
        emb=self.title_emb(title)
        return emb

