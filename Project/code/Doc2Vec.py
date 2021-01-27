import numpy as np
import pandas as pd
import jieba
from math import log10


DATA_DIR = '../data/'
stopwords_path = DATA_DIR + 'stopwords.txt'
worddict_path = DATA_DIR + 'worddict.txt'


class BaseDoc2Vec(object):
    def __init__(self):
        fh = open(stopwords_path, 'r', encoding= 'UTF-8')
        self.stopwords = [item.strip() for item in fh.readlines()]
        fh.close()
        jieba.load_userdict(worddict_path)

    def cut_words(self, doc):
        return [word for word in jieba.cut(doc) if not word in self.stopwords]



class TFIDF(BaseDoc2Vec):
    def __init__(self, docs):
        BaseDoc2Vec.__init__(self)
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

        #self.wordindex = dict([(self.words[i], i) for i in range(len(self.words))])
        print(len(docs), 'articles loaded, with word bag length:', len(self.words)) 


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


