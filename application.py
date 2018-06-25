# -*- coding: utf-8 -*-

from collections import OrderedDict
from os import path
from os.path import join as pj

import graphene as gp
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastai.column_data import ColumnarModelData
from fastai.learner import fit, get_cv_idxs, optim, set_lrs
from flask import Flask
from flask_graphql import GraphQLView
from torch import nn
from torch.autograd import Variable


class Query(gp.ObjectType):

    predicted_ratings = gp.Field(
        gp.List(gp.Float),
        users=gp.NonNull(gp.List(gp.Int)),
        books=gp.NonNull(gp.List(gp.Int)))

    def resolve_predicted_ratings(self, info, users, books):
        if len(set([len(users), len(books)])) != 1:
            return []

        with open('model-params.conf', 'r') as conf_file:
            n_users = int(conf_file.readline().strip())
            n_books = int(conf_file.readline().strip())
        N_FACTORS = 50

        model = EmbeddingDot(n_users, n_books, N_FACTORS)
        model.load_state_dict(torch.load('bookweb-embed-dot.pth'))

        data_in = pd.DataFrame.from_dict(
            OrderedDict([('userID', users), ('bookID', books)]))

        data_tsr = torch.LongTensor(data_in.values)
        data_var = Variable(data_tsr, volatile=True)

        return model(data_var, None).data.numpy().tolist()


class Retrain(gp.Mutation):
    class Arguments:
        users = gp.NonNull(gp.List(gp.Int))
        books = gp.NonNull(gp.List(gp.Int))
        ratings = gp.NonNull(gp.List(gp.Int))

    ok = gp.Boolean()

    def mutate(self, info, users, books, ratings):
        if len(set([len(users), len(books), len(ratings)])) != 1:
            return Retrain(ok=False)
        if len(users) < 10:
            return Retrain(ok=False)

        data = pd.DataFrame.from_dict({'userID': users, 'bookID': books, 'rating': ratings})

        u_uniq = data.userID.unique()
        user2idx = {o: i for i, o in enumerate(u_uniq)}
        data.userID = data.userID.apply(lambda x: user2idx[x])

        m_uniq = data.bookID.unique()
        book2idx = {o: i for i, o in enumerate(m_uniq)}
        data.bookID = data.bookID.apply(lambda x: book2idx[x])

        n_users = int(data.userID.nunique())
        n_books = int(data.bookID.nunique())

        X = data.drop(['rating'], axis=1)
        y = data['rating'].astype(np.float32)

        val_idxs = get_cv_idxs(len(data))
        model_data = ColumnarModelData.from_data_frame(
            path, val_idxs, X, y, ['userID', 'bookID'], 64)

        N_FACTORS = 50
        WD = 1e-5
        model = EmbeddingDot(n_users, n_books, N_FACTORS)
        opt = optim.SGD(
            model.parameters(), 1e-1, weight_decay=WD, momentum=0.9)

        fit(model, model_data, 20, opt, F.mse_loss)
        set_lrs(opt, 0.01)
        fit(model, model_data, 20, opt, F.mse_loss)

        torch.save(model.state_dict(), 'bookweb-embed-dot.pth')
        with open('model-params.conf', 'w') as conf_file:
            conf_file.write(f'{n_users}\n{n_books}\n')

        return Retrain(ok=True)


class Mutations(gp.ObjectType):
    retrain = Retrain.Field()


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


schema = gp.Schema(query=Query, mutation=Mutations)

application = Flask(__name__)
application.add_url_rule(
    '/', 'index',
    view_func=GraphQLView.as_view('graphql', schema=schema, graphiql=True))

if __name__ == "__main__":
    application.run()
