import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv
from torch.nn.parameter import Parameter
from layers import GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from dgl.nn.pytorch.conv import SAGEConv, APPNPConv, GINConv, SGConv, GATConv
import dgl.function as fn
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = torch.relu(x)
        # x=F.normalize(x,p=2,dim=1)
        x = self.dropout(x)
        # x=F.normalize(x,p=2,dim=1)
        x = self.gc2(x, adj)
        return 1.0*F.normalize(x,p=2,dim=1)



class GCN_T(nn.Module):
    def __init__(self, nfeat, nhid, nclass, base_model, dropout):
        super(GCN_T, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
        self.base_model = base_model
        for para in self.base_model.parameters():
            para.requires_grad = False
    def forward(self, x, adj):
        logits = self.base_model(x, adj)
        t = self.gc1(logits, adj)
        t = torch.relu(t)
        t = self.dropout(t)
        t = self.gc2(t, adj)
        t = torch.log(1.1+torch.exp(t))
        output = logits * t
        return output


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropoutvfvf

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.attentions1 = [GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att1 = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False)

        self.out_att2 = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        #
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions1], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att1(x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att2(x, adj))


        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att1 = SpGraphAttentionLayer(nhid * nheads,
                                             nhid * nheads,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

        self.out_att2 = SpGraphAttentionLayer(nhid * nheads,
                                             nhid * nheads,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att1(x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att2(x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=None))
        self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=None))
        self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=None))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features, adj):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        return h



class APPNP(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 hiddens,
                 n_classes,
                 activation,
                 dropout,
                 alpha,
                 k):
        super(APPNP, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, hiddens))

        self.layers.append(nn.Linear(hiddens, hiddens))
        self.layers.append(nn.Linear(hiddens, hiddens))

        self.layers.append(nn.Linear(hiddens, n_classes))
        self.activation = activation
        self.feat_drop = nn.Dropout(dropout)
        self.propagate = APPNPConv(k, alpha, dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features, adj):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))

        h = self.activation(self.layers[1](h))
        # h = self.activation(self.layers[2](h))

        h = self.layers[-2](self.feat_drop(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(self.g, h)
        return h

class GIN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 hidden,
                 n_classes,
                 activation,
                 feat_drop,
                 eps):
        super(GIN, self).__init__()
        self.g = g

        self.mlp1 = nn.Linear(in_feats, hidden)

        self.mlp3 = nn.Linear(hidden, hidden)

        self.mlp4 = nn.Linear(hidden, hidden)

        self.mlp2 = nn.Linear(hidden, n_classes)

        self.layer1 = GINConv(self.mlp1, 'sum', eps)
        self.layer2 = GINConv(self.mlp2, 'sum', eps)

        self.layer3 = GINConv(self.mlp3, 'sum', eps)

        self.layer4 = GINConv(self.mlp4, 'sum', eps)

        self.activation = activation

        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

    def forward(self, features, adj):
        # prediction step
        h = features
        h = self.feat_drop(h)

        h = self.layer1(self.g, h)
        h = self.activation(h)
        h = self.feat_drop(h)

        h = self.layer3(self.g, h)
        h = self.activation(h)
        h = self.feat_drop(h)

        h = self.layer4(self.g, h)
        h = self.activation(h)
        h = self.feat_drop(h)

        h = self.layer2(self.g, h)
        return h

class SGC(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 num_k):
        super(SGC, self).__init__()
        self.g = g

        self.model = SGConv(in_feats,
                   n_classes,
                   k=num_k,
                   cached=True)
        self.model1 = SGConv(in_feats,
                   n_classes,
                   k=num_k,
                   cached=True)


    def forward(self, features, adj):
        # prediction step
        h = self.model(self.g, features)

        return h
