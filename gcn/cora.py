import csv
import numpy as np
import networkx as nx
import scipy.sparse as sp
import random

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def cnt_label(mat):
    for i in range(mat.shape[1]):
        cnt=0
        for x in mat:
            if x[i]==1:
                cnt+=1
        print(i, cnt, end=',')
    print()

def dataset():
    i=0
    dicex={}
    labels=[]
    f=open("data/titles_class1.csv")
    lab=csv.reader(f)
    for l in lab:
        x=np.zeros(7)
        x[int(l[2])]=1
        labels.append(x)
        dicex[int(l[0])-1]=i
        i+=1
    f.close()
    labels=np.array(labels)

    diccite={}
    i=0
    f=open("data/cites.csv")
    cites=csv.reader(f)
    for x in cites:
        if i not in dicex:
            i+=1
            continue
        diccite[dicex[i]]=[]
        for z in x:
            zi=int(z)
            if zi in dicex:
                diccite[dicex[i]].append(dicex[zi])
        i+=1
    f.close()

    dicdocidx={}
    i=0
    f=open("data/docsx.csv")
    docsx=csv.reader(f)
    for x in docsx:
        if i in dicex:
            dicdocidx[dicex[i]]=x
        i+=1
    f.close()

    dicdoccnt={}
    i=0
    f=open("data/docsy.csv")
    docsy=csv.reader(f)
    for x in docsy:
        if i in dicex:
            x=[float(j) for j in x]
            dicdoccnt[dicex[i]]=[j/sum(x) for j in x]
        i+=1
    f.close()


    dicvocab={}
    f=open("data/vocab.csv")
    vocab=csv.reader(f)
    for word in vocab:
        if word[0]!='':
            dicvocab[str(int(word[0])-1)]=word[1]
    f.close()

    adj=nx.adj_matrix(nx.from_dict_of_lists(diccite))


    features=sp.lil_matrix((len(dicdoccnt), len(dicvocab)))
    for i in dicdocidx:
        for j in range(len(dicdocidx[i])):
            features[int(i), int(dicdocidx[i][j])]=dicdoccnt[i][j]

    ltest=1510
    lval=1200

    test_idx_range=list(range(ltest, len(dicdoccnt)))
    test_idx_reorder=list(range(ltest, len(dicdoccnt)))
    random.shuffle(test_idx_reorder)

    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range
    idx_train = range(lval)
    idx_val = range(lval, ltest)
    print(ltest)
    print(lval)


    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    cnt_label(labels)
    cnt_label(y_train)
    cnt_label(y_test)
    cnt_label(y_val)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask