# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

import graphene as gp


class Query(gp.ObjectType):

    predicted_ratings = gp.Field(gp.List(gp.Float),
                                 users=gp.NonNull(gp.List(gp.Int)),
                                 books=gp.NonNull(gp.List(gp.Int)))

    def resolve_predicted_ratings(self, info, users, books):
        data_in = pd.DataFrame({'userID': users, 'bookID': books})
        data_in = data_in.astype({'userID': 'category', 'bookID': 'category'})

        n_users = 100
        n_books = 360

        N_FACTORS = 50
        model = EmbeddingDot(n_users, n_books, N_FACTORS)
        #TODO: load weights

        data_tsr = torch.LongTensor(data_in.as_matrix())
        data_var = Variable(data_tsr, volatile=True)

        return model(data_var, None).data.numpy().tolist()


class EmbeddingDot(nn.Module):
    """ A simple dot product model """

    def __init__(self, n_users, n_books, n_factors):
        super().__init__()
        self.u = nn.Embedding(n_users, n_factors)
        self.b = nn.Embedding(n_books, n_factors)
        self.u.weight.data.uniform_(0, 0.05)
        self.b.weight.data.uniform_(0, 0.05)
        
    def forward(self, cats, conts):
        users, books = cats[:, 0], cats[:, 1]
        u, b = self.u(users), self.b(books)
        return (u * b).sum(1)


schema = gp.Schema(query=Query)