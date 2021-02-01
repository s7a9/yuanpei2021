import logging
from math import log10

import jieba
import numpy as np
import pandas as pd
from NNModules import *

DATA_DIR = '../data/'
stopwords_path = DATA_DIR + 'stopwords.txt'
worddict_path = DATA_DIR + 'worddict.txt'


class BaseDoc2Vec(object):
    """Base class for doc2vec convertor

    Provide shared functions such as cutting sentences
    """
    def __init__(self):
        fh = open(stopwords_path, 'r', encoding= 'UTF-8')
        self.stopwords = [item[:-1] for item in fh.readlines()]
        fh.close()
        jieba.load_userdict(worddict_path)
    
    def cut_words(self, doc):
        """Cut a sentence using module jieba

        Args: 
            doc : string
                sentence to cut
        Return:
            list of words : [string]
                seperate words sotred in list
        """
        return [word for word in jieba.cut(doc) if not word in self.stopwords]

class TFIDF(BaseDoc2Vec):
    """使用TF-IDF方法计算文档向量
    """
    #additional_stopwords = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    additional_stopwords = []
    
    def __init__(self, docs, freq_threshold= 2):
        """Initialize class and cut sentences

        Args:
            docs : [string]
                list of sentences
            freq_threshold : int
                the threshlod of frequency of recorded words
        """
        BaseDoc2Vec.__init__(self)  # initialize variables
        self.stopwords += self.additional_stopwords
        self.docs = []
        self.words = set()

        for doc in docs: # go through documents to record all words
            words = set()
            for word in self.cut_words(doc):
                self.words.add(word)
                words.add(word)
            self.docs.append(list(words))
        self.words = list(self.words)

        self.dfdict = dict([(wrd, 0) for wrd in self.words])
        for doc in self.docs: 
            for word in doc:
                self.dfdict[word] += 1 # calculate word frequency
        
        # exclude words that appear less than threshold
        self.words = [word for word in self.words if self.dfdict[word] > freq_threshold]
        logging.info(f'{len(docs)} articles loaded, with word bag length: {len(self.words)}')

    def doc2vec(self, doc):
        """Turn a sentence into vector

        Args:
            doc : string
                the very sentence to be converted into vector
        
        Return:
            vector : [float]
                calculated vector of sentence
        """
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
    """Neural network model for doc2vec convertor

    CBOW or Skip-gram. Load model from file is supported
    """
    additional_stopwords = []

    def __init__(self, docs, dict_path= 'wordindex.npy'):
        """Initialize module and store information of words

        Args:
            docs : [string]
                originial input texts
            dict_path : string (path)
                store the index of each words for future loading in case
                every time the result of loading original data is different.
        """
        super(NNModel, self).__init__()
        self.stopwords += self.additional_stopwords
        self.words = set(['OOB', 'UNK']) # OOB for out of boundary, UNK for unknown words
        self.docs = []

        for doc in docs:
            datum = []
            for word in self.cut_words(doc):
                self.words.add(word)
                datum.append(word)
            self.docs.append(datum)

        self.words = list(self.words)
        self.word2idx = dict([(self.words[i], i) for i in range(len(self.words))])
        logging.info(f'{len(docs)} articles loaded, with word bag length: {len(self.words)}')
        if dict_path != '': # save dict
            np.save(DATA_DIR + dict_path, self.word2idx)

    def prepare_data(self, context_size, model_name):
        """Preprocess raw text data

        Args:
            context_size : int
                the size of context used for padding data
            model_name : string ('cbow' or 'skipgram')
                generate different data for different models
        """
        self.context_size = context_size
        data_x = []
        data_y = []
        oob = self.word2idx['OOB']

        for item in self.docs:
            data = [oob] * context_size + self.doc2token(item) + [oob] * context_size #padding
            for i in range(context_size, len(data) - context_size):
                data_x.append(data[i - context_size: i] + data[i + 1: i + context_size + 1])
                data_y.append(data[i])
        
        if model_name.lower() == 'skipgram':
            data_x, data_y = data_y, data_x
        self.data_x = Variable(torch.LongTensor(data_x))
        self.data_y = Variable(torch.LongTensor(data_y))
        logging.info(f'data preprocessed, data shape: {self.data_x.shape}, {self.data_y.shape}')

    def doc2token(self, doc):
        """Get list of tokens accoring to input text (already cut)

        Args:
            list of words : [string]
                Input text (already cut)
        Return:
            list of tokens : [int]
                Tokens of input text
        """
        return [self.word2idx[word] if self.word2idx.__contains__(word)
                else self.word2idx['UNK'] for word in doc]
    
    def train_model(self, nnModel, epoch_n, embed_size, 
                    batch_size= 200, lr= 1e-3, continue_train= False):
        """Train the model

        Args:
            nnModel : NN Module class
                The NN module to train (CBOW or SkipGram)
            epoch_n : int
                # of epoch for training
            embed_size : int
                Finial size of output vector
            batch_size : int
                size of one batch
            lr : float
                learning rate
            continue_train : bool
                whether training is based on previously saved NN
        """
        if continue_train:
            model = self.model.train()
        else:
            model = nnModel(len(self.word2idx), embed_size, self.context_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr= lr)
        if torch.cuda.is_available():
            model = model.cuda()
        dataset = Data.TensorDataset(self.data_x, self.data_y)
        data_loader = Data.DataLoader(dataset= dataset,
                                      batch_size= batch_size, 
                                      shuffle= True)
        logging.info('Start training...')
        for epoch in range(epoch_n):
            for step, (batch_x, batch_y) in enumerate(data_loader):
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                pred_y = model(batch_x)
                loss = criterion(pred_y, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                del batch_x
                del batch_y
                del pred_y
            if (epoch + 1) % 50 == 0:
                logging.info(f'Batch {epoch + 1}\tLoss {loss.item()}')
                torch.save(model.state_dict(), DATA_DIR + f'savedmodel/{epoch + 1}_{loss.item()}.model')
        self.model = model.cpu()
    
    def load_model(self, dict_path, path, nnModel, context_size, embed_size):
        """load previously trained NN

        Args:
            dict_path : string (path)
                path of word to token dictionry
            path : string (path)
                path of saved NN module
            nnModel : NN Module class
                NN module trained
            context_size : int
                size of context
            embed_size : int
                size of output vector
        """
        self.word2idx = np.load(dict_path, allow_pickle= True).item()
        self.model = nnModel(len(self.word2idx), embed_size, context_size)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info(f'model loaded, and word dict len: {len(self.word2idx)}')
    
    def word2vec(self, words):
        """Get vector according to words

        Args:
            words : [string]
                list of words
        Return:
            vector : numpy.array
                vector of words, shape: (# of words, embed_size)
        """
        with torch.no_grad():
            words = torch.LongTensor(self.doc2token(words))
            result = self.model.embedding(words).numpy()
        return result
