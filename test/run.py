from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
from utils import *

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_recall_fscore_support

# args
data_name = 'Cora' # from Cora or Pubmed
seed = 42
fcn_node = True
fcn_link = True
gcn_node = True
gcn_link = True
gra_sage = True
node_vec = True
strict = False

# LogisticRegressionCV args
solver = 'sag'
n_jobs = 1
max_iter = 1000

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if fcn_node:
    print('Node classification by FCN on {}'.format(data_name))
    from fcn import TwoLayerFCN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid('./data', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    num_nodes, num_features = data.x.size()
    num_classes = len(data.y.unique())

    # one hot encode labels
    y = F.one_hot(data.y)
    y_tr, y_val, y_te = y[data.train_mask], y[data.val_mask], y[data.test_mask]
    y_tr = torch.cat([y_tr, torch.zeros(size=(num_nodes - y_tr.shape[0], num_classes))], axis=0)
    y_val = torch.cat([y_val, torch.zeros(size=(num_nodes - y_val.shape[0], num_classes))], axis=0)
    y_te = torch.cat([y_te, torch.zeros(size=(num_nodes - y_te.shape[0], num_classes))], axis=0)

    # fix shapes of data
    X = data.x
    X_tr, X_val, X_te = X[data.train_mask], X[data.val_mask], X[data.test_mask]
    X_tr = torch.cat([X_tr, torch.zeros(size=(num_nodes - X_tr.shape[0], num_features))], axis=0)
    X_val = torch.cat([X_val, torch.zeros(size=(num_nodes - X_val.shape[0], num_features))], axis=0)
    X_te = torch.cat([X_te, torch.zeros(size=(num_nodes - X_te.shape[0], num_features))], axis=0)

    tr_shuffle, val_shuffle, te_shuffle = torch.randperm(num_nodes), torch.randperm(num_nodes), torch.randperm(num_nodes)
    X_tr, X_val, X_te = X_tr[tr_shuffle], X_val[val_shuffle], X_te[te_shuffle]

    lr = 1e-2
    epochs = 400
    print_every = 20

    model = TwoLayerFCN(num_classes, num_features, num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for t in range(1, epochs + 1):
        model.train()
        shuffle = torch.randperm(num_nodes)
        shuffle_mask = data.train_mask[shuffle]
        
        inputs, train_targets = X_tr[shuffle].to(device), y_tr[shuffle].to(device)
        train_preds = model(inputs)
        train_loss = loss_fn(train_preds[shuffle_mask], train_targets[shuffle_mask])
        train_acc = (train_preds.argmax(axis=1) == train_targets.argmax(axis=1)).float().mean()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            shuffle = torch.randperm(num_nodes)
            shuffle_mask = data.val_mask[shuffle]

            inputs, val_targets = X_val[shuffle].to(device), y_val[shuffle].to(device)
            val_preds = model(inputs)
            val_loss = loss_fn(val_preds[shuffle_mask], val_targets[shuffle_mask])
            val_acc = (val_preds.argmax(axis=1) == val_targets.argmax(axis=1)).float().mean()

        if t % print_every == 0 or t == 1: print(
            'Epoch {:3d}'.format(t) 
            + ' | tr xe: {:.6f}; val xe: {:.6f};'.format(train_loss.item(), val_loss.item())
            + ' tr acc: {:.6f}; val acc: {:.6f}'.format(train_acc.item(), val_acc.item())
        )

    inputs, tr_targets = X_tr.to(device), y_tr.to(device)
    tr_preds = model(inputs)
    tr_acc = (tr_preds.argmax(axis=1) == tr_targets.argmax(axis=1)).float().mean().item()

    inputs, te_targets = X_te.to(device), y_te.to(device)
    te_preds = model(inputs)
    te_acc = (te_preds.argmax(axis=1) == te_targets.argmax(axis=1)).float().mean().item()
    
    print('tr acc: {:.4f}'.format(tr_acc))
    print('te acc: {:.4f}'.format(te_acc))

if fcn_link:
    print('Link prediction by FCNEdge on {}'.format(data_name))
    from fcn_edge import TwoLayerFCNEdge

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid('./data', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    num_nodes, num_features = data.x.size()
    X = data.x
    A = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.shape[1])).to_dense()

    # separate into train, val, test
    X_tr, X_val, X_te = X[data.train_mask], X[data.val_mask], X[data.test_mask]
    A_tr, A_val, A_te = A[data.train_mask], A[data.val_mask], A[data.test_mask]

    # get indices of train, val, test
    mask_tr = torch.cat([torch.ones(len(X_tr)), torch.zeros(num_nodes - len(X_tr))]).bool()
    mask_val = torch.cat([torch.ones(len(X_val)), torch.zeros(num_nodes - len(X_val))]).bool()
    mask_te = torch.cat([torch.ones(len(X_te)), torch.zeros(num_nodes - len(X_te))]).bool()

    # fix shapes
    X_tr = torch.cat([X_tr, torch.zeros(size=(num_nodes - X_tr.shape[0], num_features))], axis=0)
    A_tr = torch.cat([A_tr, torch.zeros(size=(num_nodes - A_tr.shape[0], num_nodes))], axis=0)

    X_val = torch.cat([X_val, torch.zeros(size=(num_nodes - X_val.shape[0], num_features))], axis=0)
    A_val = torch.cat([A_val, torch.zeros(size=(num_nodes - A_val.shape[0], num_nodes))], axis=0)

    X_te = torch.cat([X_te, torch.zeros(size=(num_nodes - X_te.shape[0], num_features))], axis=0)
    A_te = torch.cat([A_te, torch.zeros(size=(num_nodes - A_te.shape[0], num_nodes))], axis=0)

    # permute
    shuffle_tr, shuffle_val, shuffle_te = torch.randperm(num_nodes), torch.randperm(num_nodes), torch.randperm(num_nodes)
    X_tr, X_val, X_te = X_tr[shuffle_tr], X_val[shuffle_val], X_te[shuffle_te]
    A_tr, A_val, A_te = A_tr[shuffle_tr], A_val[shuffle_val], A_te[shuffle_te]
    mask_tr, mask_val, mask_te = mask_tr[shuffle_tr], mask_val[shuffle_val], mask_te[shuffle_te]

    lr = 1e-4
    if data_name == 'Cora':
        epochs = 600
        print_every = 15
    elif data_name == 'Pubmed':
        epochs = 200
        print_every = 5

    model = TwoLayerFCNEdge(num_nodes, num_features, 64, 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for t in range(1, epochs + 1):
        model.train()
        shuffle = torch.randperm(num_nodes)
        input, targets, shuffle_mask = X_tr[shuffle].to(device), A_tr[shuffle].to(device), mask_tr[shuffle]
        preds = model(input)
        train_loss = loss_fn(preds[mask_tr], targets[mask_tr])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            shuffle = torch.randperm(num_nodes)

            inputs, targets, shuffle_mask = X_tr[shuffle].to(device), A_tr[shuffle].to(device), mask_tr[shuffle]
            preds = model(inputs)
            tr_prec, tr_rcll = report(targets, preds, shuffle_mask)

            input, targets, shuffle_mask = X_val[shuffle].to(device), A_val[shuffle].to(device), mask_val[shuffle]
            preds = model(input)
            val_loss = loss_fn(preds[shuffle_mask], targets[shuffle_mask])
            val_prec, val_rcll = report(targets, preds, shuffle_mask)

        if t % print_every == 0 or t == 1: print(
            'Epoch {:3d} | '.format(t) 
            + 'tr xe: {:.4f}; val xe: {:.4f}; '.format(train_loss.item(), val_loss.item())
            + 'tr prec: {:.4f}; val prec: {:.4f}; '.format(tr_prec, val_prec) 
            + 'tr recall: {:.4f}; val recall: {:.4f}'.format(tr_rcll, tr_rcll)
        )
    with torch.no_grad():
        model.eval()
        inputs, targets = X_tr.to(device), A_tr.to(device)
        preds = model(inputs)
        tr_prec, tr_recall = report(targets, preds, mask_tr)

        inputs, targets = X_te.to(device), A_te.to(device)
        preds = model(inputs)
        te_prec, te_recall = report(targets, preds, mask_te)
        print('tr prec: {:.4f}; tr recall: {:.4f}'.format(tr_prec, tr_recall))
        print('te prec: {:.4f}; te recall: {:.4f}'.format(te_prec, te_recall))

if gcn_node and data_name == 'Cora':
    print('Node classification by GCN on {}'.format(data_name))

    from gcn import TwoLayerGCN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid('./data', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]
    num_nodes, num_features = data.x.size()
    num_classes = len(data.y.unique())

    X = data.x
    y = F.one_hot(data.y)

    # separate train, val, test
    X_tr, X_val, X_te = X[data.train_mask], X[data.val_mask], X[data.test_mask]
    y_tr, y_val, y_te = y[data.train_mask], y[data.val_mask], y[data.test_mask]

    # get indices of train, val, test
    mask_tr = torch.cat([torch.ones(len(X_tr)), torch.zeros(num_nodes - len(X_tr))]).bool()
    mask_val = torch.cat([torch.ones(len(X_val)), torch.zeros(num_nodes - len(X_val))]).bool()
    mask_te = torch.cat([torch.ones(len(X_te)), torch.zeros(num_nodes - len(X_te))]).bool()

    # fix shapes
    X_tr = torch.cat([X_tr, torch.zeros(size=(num_nodes - X_tr.shape[0], num_features))])
    y_tr = torch.cat([y_tr, -torch.ones(size=(num_nodes - y_tr.shape[0], num_classes))])

    X_val = torch.cat([X_val, torch.zeros(size=(num_nodes - X_val.shape[0], num_features))])
    y_val = torch.cat([y_val, -torch.ones(size=(num_nodes - y_val.shape[0], num_classes))])

    X_te = torch.cat([X_te, torch.zeros(size=(num_nodes - X_te.shape[0], num_features))])
    y_te = torch.cat([y_te, -torch.ones(size=(num_nodes - y_te.shape[0], num_classes))])

    torch.manual_seed(41)

    # get permutations
    shuffle_tr, shuffle_val, shuffle_te = torch.randperm(num_nodes), torch.randperm(num_nodes), torch.randperm(num_nodes)
    X_tr, X_val, X_te = X_tr[shuffle_tr], X_val[shuffle_val], X_te[shuffle_te]
    y_tr, y_val, y_te = y_tr[shuffle_tr], y_val[shuffle_val], y_te[shuffle_te]
    mask_tr, mask_val, mask_te = mask_tr[shuffle_tr], mask_val[shuffle_val], mask_te[shuffle_te]

    # set training parameters
    lr = 1e-4
    epochs = 300
    print_every = 15

    A = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.shape[1]))
    model = TwoLayerGCN(A, num_classes, num_features, p=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for t in range(1, epochs + 1):
        model.train()
        shuffle = torch.randperm(num_nodes)
        
        inputs, targets, shuffle_mask = X_tr[shuffle].to(device), y_tr[shuffle].to(device), mask_tr[shuffle]
        preds = model(inputs)
        train_loss = loss_fn(preds[shuffle_mask], targets[shuffle_mask])
        train_acc = (
            preds[shuffle_mask].argmax(axis=1) == targets[shuffle_mask].argmax(axis=1)
        ).float().mean()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            shuffle = torch.randperm(num_nodes)

            inputs, targets, shuffle_mask = X_val[shuffle].to(device), y_val[shuffle].to(device), mask_val[shuffle]
            preds = model(inputs)
            val_loss = loss_fn(preds[shuffle_mask], targets[shuffle_mask])
            val_acc = (
                preds[shuffle_mask].argmax(axis=1) == targets[shuffle_mask].argmax(axis=1)
            ).float().mean()

        if t % print_every == 0 or t == 1: print(
            'Epoch {:3d} | '.format(t) 
            + 'tr xe: {:.6f}; val xe: {:.6f}; '.format(train_loss.item(), val_loss.item())
            + 'tr acc: {:.4f}; val acc: {:.4f}'.format(train_acc.item(), val_acc.item())
        )
    tr_acc = (model(X_tr)[mask_tr].argmax(axis=1) == y_tr[mask_tr].argmax(axis=1)).float().mean()
    te_acc = (model(X_te)[mask_te].argmax(axis=1) == y_te[mask_te].argmax(axis=1)).float().mean()
    print('tr acc: {:4f}'.format(tr_acc.item()))
    print('te acc: {:4f}'.format(te_acc.item()))

if gcn_link and data_name == 'Cora':
    print('Link prediction by GCNEdge on {}'.format(data_name))
    from gcn_edge import TwoLayerGCNEdge

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid('./data', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]
    num_nodes, num_features = data.x.size()

    X = data.x
    A = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.shape[1])).to_dense()

    # separate into train, val, test
    X_tr, X_val, X_te = X[data.train_mask], X[data.val_mask], X[data.test_mask]
    A_tr, A_val, A_te = A[data.train_mask], A[data.val_mask], A[data.test_mask]

    # get indices of train, val, test
    mask_tr = torch.cat([torch.ones(len(X_tr)), torch.zeros(num_nodes - len(X_tr))]).bool()
    mask_val = torch.cat([torch.ones(len(X_val)), torch.zeros(num_nodes - len(X_val))]).bool()
    mask_te = torch.cat([torch.ones(len(X_te)), torch.zeros(num_nodes - len(X_te))]).bool()

    # fix shapes
    X_tr = torch.cat([X_tr, torch.zeros(size=(num_nodes - X_tr.shape[0], num_features))], axis=0)
    A_tr = torch.cat([A_tr, torch.zeros(size=(num_nodes - A_tr.shape[0], num_nodes))], axis=0)

    X_val = torch.cat([X_val, torch.zeros(size=(num_nodes - X_val.shape[0], num_features))], axis=0)
    A_val = torch.cat([A_val, torch.zeros(size=(num_nodes - A_val.shape[0], num_nodes))], axis=0)

    X_te = torch.cat([X_te, torch.zeros(size=(num_nodes - X_te.shape[0], num_features))], axis=0)
    A_te = torch.cat([A_te, torch.zeros(size=(num_nodes - A_te.shape[0], num_nodes))], axis=0)

    # permute
    shuffle_tr, shuffle_val, shuffle_te = torch.randperm(num_nodes), torch.randperm(num_nodes), torch.randperm(num_nodes)
    X_tr, X_val, X_te = X_tr[shuffle_tr], X_val[shuffle_val], X_te[shuffle_te]
    A_tr, A_val, A_te = A_tr[shuffle_tr], A_val[shuffle_val], A_te[shuffle_te]
    mask_tr, mask_val, mask_te = mask_tr[shuffle_tr], mask_val[shuffle_val], mask_te[shuffle_te]

    lr = 1e-12 # somehow 1e-12 works well
    epochs = 500
    print_every = 20

    model = TwoLayerGCNEdge(A_tr, num_features, 64, 16, p=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for t in range(1, epochs + 1):
        model.train()
        shuffle = torch.randperm(num_nodes)
        inputs, targets, shuffle_mask = X_tr[shuffle].to(device), A_tr[shuffle].to(device), mask_tr[shuffle]
        preds = model(inputs)
        train_loss = loss_fn(preds[shuffle_mask], targets[shuffle_mask])

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            shuffle = torch.randperm(num_nodes)

            inputs, targets, shuffle_mask = X_tr[shuffle].to(device), A_tr[shuffle].to(device), mask_tr[shuffle]
            preds = model(inputs)
            train_prec, train_rcll = report(targets, preds, shuffle_mask)

            inputs, targets, shuffle_mask = X_val[shuffle].to(device), A_val[shuffle].to(device), mask_val[shuffle]
            preds = model(inputs)
            val_loss = loss_fn(preds[shuffle_mask], targets[shuffle_mask])
            val_prec, val_rcll = report(targets, preds, shuffle_mask)

        if t % print_every == 0 or t == 1: print(
            'Epoch {:3d} | '.format(t) 
            + 'tr xe: {:.4f}; val xe: {:.4f}; '.format(train_loss.item(), val_loss.item())
            + 'tr prec: {:.4f}; val prec: {:.4f}; '.format(train_prec, val_prec)
            + 'tr recall: {:.4f}; val recall: {:.4f}'.format(train_rcll, val_rcll)
        )
    with torch.no_grad():
        model.eval()
        inputs, targets = X_tr.to(device), A_tr.to(device)
        preds = model(inputs)
        tr_prec, tr_recall = report(targets, preds, mask_tr)

        inputs, targets = X_te.to(device), A_te.to(device)
        preds = model(inputs)
        te_prec, te_recall = report(targets, preds, mask_te)
        print('tr prec: {:.4f}; tr recall: {:.4f}'.format(tr_prec, tr_recall))
        print('te prec: {:.4f}; te recall: {:.4f}'.format(te_prec, te_recall))

if gra_sage:
    from torch.autograd import Variable
    from graphsage.encoders import Encoder
    from graphsage.aggregators import MeanAggregator
    from graphsage.spv_graphsage import SupervisedGraphSage

    if data_name == 'Cora':
        from graphsage.model import load_cora
        num_nodes, num_feats, num_classes = 2708, 1433, 7
        feat_data, labels, adj_lists = load_cora()
        indices = np.array(range(num_nodes))
        train = list(indices[:140])
        val = indices[140:640]
        test = indices[-1000:]

    elif data_name == 'Pubmed':
        from graphsage.model import load_pubmed
        num_nodes, num_feats, num_classes = 19717, 500, 3
        feat_data, labels, adj_lists = load_pubmed()
        indices = np.array(range(num_nodes))
        train = list(indices[:60])
        val = indices[60:560]
        test = indices[18717:]
    else:
        print('GraphSage is not implemented for {}.'.format(data_name))

    # initialize model
    features = nn.Embedding(num_nodes, num_feats)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, num_feats, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(
            lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples, enc2.num_samples = 5, 5
    graphsage = SupervisedGraphSage(num_classes, enc2)

    print('Training GraphSage on {}'.format(data_name))
    batch_size = 32
    optimizer = torch.optim.Adam(graphsage.parameters(), lr=1e-3)
    print_every = 25
    epochs = 500
    for batch in range(1, epochs + 1):
        batch_nodes = train[:batch_size]
        random.shuffle(train)
        optimizer.zero_grad()
        loss = graphsage.loss(
            batch_nodes, 
            Variable(torch.LongTensor(labels[np.array(batch_nodes)]))
        )
        loss.backward()
        optimizer.step()
        if batch == 1 or batch % print_every == 0:
            print('batch {} | xe: {:.6f}'.format(batch, loss.data.item()))

    tr_output = graphsage.forward(train)
    val_output = graphsage.forward(val) 

    print('Test quality of GraphSage embeddings for node classification on {}.'.format(data_name))
    X_tr = graphsage.enc(train).t().detach().numpy()
    X_te = graphsage.enc(test).t().detach().numpy()
    y_tr = labels[np.array(train)].squeeze()
    y_te = labels[np.array(test)].squeeze()

    clf = LogisticRegressionCV(class_weight='balanced', max_iter=max_iter).fit(X_tr, y_tr)
    tr_acc, te_acc = clf.score(X_tr, y_tr), clf.score(X_te, y_te)
    print('tr acc: {:.4f}'.format(tr_acc))
    print('te acc: {:.4f}'.format(te_acc))

    print('Test quality of GraphSage embeddings for link prediction on {}'.format(data_name))
    dataset = Planetoid('./data', data_name)
    data = dataset[0]
    feat_data = data.x
    adj_lists = data.edge_index
    num_nodes, num_features = feat_data.shape

    print('Generating existing and nonexistent edges.')
    pos_samples_tr, neg_samples_tr = sample_edges(adj_lists, data.train_mask, strict=strict)
    pos_samples_te, neg_samples_te = sample_edges(adj_lists, data.test_mask, strict=strict)

    # use pointwise multiplication as aggregator
    node_embeddings = graphsage.enc(list(range(num_nodes))).t().detach()
    embeddings_pos_hdmd_tr = node_embeddings[pos_samples_tr, :][:, 0, :] * node_embeddings[pos_samples_tr, :][:, 1, :]
    embeddings_neg_hdmd_tr = node_embeddings[neg_samples_tr, :][:, 0, :] * node_embeddings[neg_samples_tr, :][:, 1, :]
    embeddings_pos_hdmd_te = node_embeddings[pos_samples_te, :][:, 0, :] * node_embeddings[pos_samples_te, :][:, 1, :]
    embeddings_neg_hdmd_te = node_embeddings[neg_samples_te, :][:, 0, :] * node_embeddings[neg_samples_te, :][:, 1, :]

    # construct train and test sets
    embeddings_hdmd_tr = np.concatenate([
        embeddings_pos_hdmd_tr, 
        embeddings_neg_hdmd_tr
    ])

    targets_tr = np.concatenate([
        np.ones(len(embeddings_pos_hdmd_tr)),
        np.zeros(len(embeddings_neg_hdmd_tr)),
    ])

    embeddings_hdmd_te = np.concatenate([
        embeddings_pos_hdmd_te,
        embeddings_neg_hdmd_te
    ])

    targets_te = np.concatenate([
        np.ones(len(embeddings_pos_hdmd_te)),
        np.zeros(len(embeddings_neg_hdmd_te))
    ])

    # finally use logistic regression
    clf = LogisticRegressionCV(class_weight='balanced', max_iter=max_iter).fit(embeddings_hdmd_tr, targets_tr)
    tr_outputs = clf.predict(embeddings_hdmd_tr)
    te_outputs = clf.predict(embeddings_hdmd_te)

    tr_prec, tr_recall, _, _ = precision_recall_fscore_support(targets_tr, tr_outputs, average="micro")
    te_prec, te_recall, _, _ = precision_recall_fscore_support(targets_te, te_outputs, average="micro")
    print('tr prec: {:.4f}; tr recall: {:.4f}'.format(tr_prec, tr_recall))
    print('te prec: {:.4f}; te recall: {:.4f}'.format(te_prec, te_recall))

if node_vec:
    from torch_geometric.nn import Node2Vec

    dataset = Planetoid('./data', data_name)
    data = dataset[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    node2vec = Node2Vec(
        data.edge_index, embedding_dim=128, walk_length=20,
        context_size=10, walks_per_node=10,
        num_negative_samples=1, p=1, q=1, sparse=True
    ).to(device)

    loader = node2vec.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    if data_name == 'Cora':
        epochs = 100
    elif data_name == 'Pubmed':
        epochs = 20
    else:
        print('Not implemented for {}'.format(data_name))
    print_every = 5

    print('Training node2vec on {}'.format(data_name))
    for t in range(1, epochs + 1):
        # train
        node2vec.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss / len(loader)
        
        # val
        node2vec.eval()
        node_embeddings = node2vec()
        acc = node2vec.test(
            node_embeddings[data.train_mask], data.y[data.train_mask],
            node_embeddings[data.val_mask], data.y[data.val_mask],
            max_iter = 150
        )
        if t == 1 or t % print_every == 0:
            print('Epoch {:3d} | Log-likelihood: {:.6f}; val acc: {:.4f}'.format(t, loss, acc))
    node_embeddings = node2vec()

    print('Test quality of node2vec embeddings for node classification on {}'.format(data_name))

    embeddings_tr = node_embeddings[data.train_mask].detach().cpu().numpy()
    targets_tr = data.y[data.train_mask].detach().cpu().numpy()
    embeddings_te = node_embeddings[data.test_mask].detach().cpu().numpy()
    targets_te = data.y[data.test_mask].detach().cpu().numpy()
    clf = LogisticRegressionCV(solver=solver, max_iter=max_iter, n_jobs=n_jobs)
    clf.fit(embeddings_tr, targets_tr)
    tr_acc, te_acc = clf.score(embeddings_tr, targets_tr), clf.score(embeddings_te, targets_te)
    print('tr acc: {:.4f}'.format(tr_acc))
    print('te acc: {:.4f}'.format(te_acc))

    print('Test quality of node2vec embeddings for link prediction on {}'.format(data_name))
    print('Generating existing and nonexistent edges.')
    pos_samples_tr, neg_samples_tr = sample_edges(data.edge_index, data.train_mask, strict=strict)
    pos_samples_te, neg_samples_te = sample_edges(data.edge_index, data.test_mask, strict=strict)

    # apply Hadamard as binary operator to embeddings
    embeddings_pos_hdmd_tr = node_embeddings[pos_samples_tr, :][:, 0, :] * node_embeddings[pos_samples_tr, :][:, 1, :]
    embeddings_neg_hdmd_tr = node_embeddings[neg_samples_tr, :][:, 0, :] * node_embeddings[neg_samples_tr, :][:, 1, :]
    embeddings_pos_hdmd_te = node_embeddings[pos_samples_te, :][:, 0, :] * node_embeddings[pos_samples_te, :][:, 1, :]
    embeddings_neg_hdmd_te = node_embeddings[neg_samples_te, :][:, 0, :] * node_embeddings[neg_samples_te, :][:, 1, :]

    # concatenate embeddings and targets
    embeddings_hdmd_tr = torch.cat([
        embeddings_pos_hdmd_tr, 
        embeddings_neg_hdmd_tr, 
    ]).detach().cpu().numpy()

    targets_tr = torch.cat([
        torch.ones(len(embeddings_pos_hdmd_tr)),
        torch.zeros(len(embeddings_neg_hdmd_tr)),
    ]).detach().cpu().numpy()

    embeddings_hdmd_te = torch.cat([
        embeddings_pos_hdmd_te,
        embeddings_neg_hdmd_te
    ]).detach().cpu().numpy()

    targets_te = torch.cat([
        torch.ones(len(embeddings_pos_hdmd_te)),
        torch.zeros(len(embeddings_neg_hdmd_te))
    ]).detach().cpu().numpy()

    clf = LogisticRegressionCV(class_weight='balanced', max_iter=1000, n_jobs=-1)
    clf.fit(embeddings_hdmd_tr, targets_tr)
    tr_outputs = clf.predict(embeddings_hdmd_tr)
    te_outputs = clf.predict(embeddings_hdmd_te)

    tr_prec, tr_recall, _, _ = precision_recall_fscore_support(targets_tr, tr_outputs, average='micro')
    te_prec, te_recall, _, _ = precision_recall_fscore_support(targets_te, te_outputs, average='micro')

    print('tr prec: {:.4f}; tr recall: {:.4f}'.format(tr_prec, tr_recall))
    print('te prec: {:.4f}; te recall: {:.4f}'.format(te_prec, te_recall))
