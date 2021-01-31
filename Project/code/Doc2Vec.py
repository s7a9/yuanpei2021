import logging
from math import log10

import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch import nn, optim
from torch.autograd import Variable

DATA_DIR = '../data/'
stopwords_path = DATA_DIR + 'stopwords.txt'
worddict_path = DATA_DIR + 'worddict.txt'


class BaseDoc2Vec(object):
    def __init__(self):
        fh = open(stopwords_path, 'r', encoding= 'UTF-8')
        self.stopwords = [item[:-1] for item in fh.readlines()]
        fh.close()
        jieba.load_userdict(worddict_path)

    def cut_words(self, doc):
        return [word for word in jieba.cut(doc) if not word in self.stopwords]

class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        self.linear1 = nn.Linear(2 * context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)
        self.context_size = context_size
        self.n_dim = n_dim
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, 2 * self.context_size * self.n_dim)
        x = self.linear1(x)
        x = F.relu(x, inplace= True)
        x = self.linear2(x)
        x = F.log_softmax(x, dim= 1)
        return x

class SkipGram(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(SkipGram, self).__init__()
        self.context_size = context_size
        self.n_dim = n_dim
        self.embedding = nn.Embedding(n_word, n_dim)
        self.linear1 = nn.Linear(n_dim, 128)
        self.linear2 = nn.Linear(128, context_size * n_dim)
    
    def forward(self, x):
        x = self.embedding(x).view(-1, self.n_dim)
        x = self.linear1(x)
        x = F.relu(x, inplace= True)
        x = self.linear2(x)
        x = F.log_softmax(x, dim= 1).view(-1, )
        return x

class TFIDF(BaseDoc2Vec):
    #additional_stopwords = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    additional_stopwords = []
    
    def __init__(self, docs, freq_threshold= 2):
        BaseDoc2Vec.__init__(self)
        self.stopwords += self.additional_stopwords
        self.docs = []
        self.words = set()

        for doc in docs:
            words = set()
            for word in self.cut_words(doc):
                self.words.add(word)
                words.add(word)
            self.docs.append(list(words))
        self.words = list(self.words)

        self.dfdict = dict([(wrd, 0) for wrd in self.words])
        for doc in self.docs: 
            for word in doc:
                self.dfdict[word] += 1
        
        self.words = [word for word in self.words if self.dfdict[word] > freq_threshold]
        logging.info(f'{len(docs)} articles  loaded, with word bag length: {len(self.words)}')


    def calc_doc_vec(self, doc):
        contained_words = self.cut_words(doc)
        vec = []
        for wrd in self.words:
            tf = contained_words.count(wrd) / len(contained_words)
            df = self.dfdict[wrd] + 1
            if wrd in contained_words: df += 1
            idf = log10((len(self.docs) + 1) / df)
            vec.append(tf * idf)
        return vec

class NNModel(BaseDoc2Vec):
    additional_stopwords = []

    def __init__(self, docs, dict_path= 'wordindex.npy'):
        super(NNModel, self).__init__()
        self.stopwords += self.additional_stopwords
        self.words = set(['OOB', 'UNK'])
        self.docs = []

        for doc in docs:
            datum = []
            for word in self.cut_words(doc):
                self.words.add(word)
                datum.append(word)
            self.docs.append(datum)

        self.words = list(self.words)
        self.word2idx = dict([(self.words[i], i) for i in range(len(self.words))])
        logging.info(f'{len(docs)} articles  loaded, with word bag length: {len(self.words)}')
        np.save(DATA_DIR + dict_path, self.word2idx)

    def prepare_data(self, context_size):
        self.context_size = context_size
        data_x = []
        data_y = []
        oob = self.word2idx['OOB']

        for item in self.docs:
            data = [oob] * context_size + self.doc2token(item) + [oob] * context_size #padding
            for i in range(context_size, len(data) - context_size):
                data_x.append(data[i - context_size: i] + data[i + 1: i + context_size + 1])
                data_y.append(data[i])
        
        self.data_x = Variable(torch.LongTensor(data_x))
        self.data_y = Variable(torch.LongTensor(data_y))
        logging.info(f'data preprocessed, data shape: {self.data_x.shape}, {self.data_y.shape}')

    def doc2token(self, doc):
        return [self.word2idx[word] if self.word2idx.__contains__(word)
                else self.word2idx['UNK'] for word in doc]
    
    def train_model(self, nnModel, epoch_n, embed_size, 
                    batch_size= 200, lr= 1e-3, continue_train= False):
        if continue_train:
            model = self.model.train()
        else:
            model = nnModel(len(self.word2idx), embed_size, self.context_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr= lr)
        if torch.cuda.is_available():
            model = model.cuda()
            data_x = self.data_x.cuda()
            data_y = self.data_y.cuda()
        else:
            data_x = self.data_x
            data_y = self.data_y
        dataset = Data.TensorDataset(data_x, data_y)
        data_loader = Data.DataLoader(dataset= dataset,
            batch_size= batch_size,
            shuffle= True)
        
        logging.info('Start training...')
        for epoch in range(epoch_n):
            for step, (batch_x, batch_y) in enumerate(data_loader):
                pred_y = model(batch_x)
                loss = criterion(pred_y, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 50 == 0:
                logging.info(f'Batch {epoch + 1}\tLoss {loss.item()}')
                torch.save(model.state_dict(), DATA_DIR + f'savedmodel/{epoch + 1}_{loss.item()}.model')
            
        self.model = model.cpu()
    
    def load_model(self, dict_path, path, nnModel, context_size, embed_size):
        self.word2idx = np.load(dict_path, allow_pickle= True).item()
        self.model = nnModel(len(self.word2idx), embed_size, context_size)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info(f'model loaded, and word dict len: {len(self.word2idx)}')
    
    def word2vec(self, doc):
        with torch.no_grad():
            words = torch.LongTensor(self.doc2token(self.cut_words(doc)))
            result = self.model.embedding(words).numpy()
        return result
