import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch import nn, optim
from torch.autograd import Variable


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
        self.linear2 = nn.Linear(128, 2 * context_size * n_dim)
    
    def forward(self, x):
        x = self.embedding(x).view(-1, self.n_dim)
        x = self.linear1(x)
        x = F.relu(x, inplace= True)
        x = self.linear2(x)
        x = F.log_softmax(x, dim= 1).view(-1, 2 * self.context_size * self.n_dim)
        return x
