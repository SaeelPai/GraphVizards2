# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 18:45:41 2021

@author: paisa
"""

from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import keras
import csv
	
# neural network with keras 
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# these train_pos, etc data is wrong. It was made based on the incorrect
# train_pos_u, train_pos_v, etc data. Use the train_data, test_data, etc now
# comment updated on 2 Aug 2021
train_pos = np.load('dummy_data/train_pos.npy')
train_neg = np.load('dummy_data/train_neg.npy')
test_pos = np.load('dummy_data/test_pos.npy')
test_neg = np.load('dummy_data/test_neg.npy')

# node2vec embeddings
wv = KeyedVectors.load("node_embeddings/node2vec.wordvectors", mmap='r')

with open('dummy_data/Dict_str_to_int.csv', mode='r') as inp:
    reader = csv.reader(inp)
    dict_nodes = {rows[0]:rows[1] for rows in reader}

#%% Input as dot product
dim_emb = 128;

train_input_pos = np.zeros((train_pos.shape[0],dim_emb))
for x in range(train_pos.shape[0]):
    train_input_pos[x] = np.multiply(wv[train_pos[(x,0)]],wv[train_pos[(x,1)]])
    
train_out_pos = np.ones((train_pos.shape[0],1))
train_pos_in_out = np.append(train_input_pos, train_out_pos, axis=1)

train_input_neg = np.zeros((train_neg.shape[0],dim_emb))
for x in range(train_neg.shape[0]):
    train_input_neg[x] = np.multiply(wv[train_neg[(x,0)]],wv[train_neg[(x,1)]])
    
train_out_neg = np.zeros((train_neg.shape[0],1))
train_neg_in_out = np.append(train_input_neg, train_out_neg, axis=1)

# final training data. each row is a new reading, the last column has output values (1,0)
train_data = np.concatenate((train_pos_in_out,train_neg_in_out))
np.random.shuffle(train_data)
training, val = train_data[:340000,:], train_data[340000:,:]

#%% Training an ANN

X = train_data[:,:-1];
y = train_data[:,-1];
X_t = training[:,:-1];
y_t = training[:,-1]
X_v = val[:,:-1];
y_v = val[:,-1]

model = keras.Sequential(
    [
        Dense(64, input_dim=128, activation="relu", name="layer1"),
        Dense(16, activation="relu", name="layer2"),
        Dense(1, activation="sigmoid", name="output")
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_t, y_t, epochs=50, batch_size=500)

#%% Evaluate the model

# pred = model.predict(X_v);

# score = 0;
# for i in range(len(y_v)):
#     if pred[i] >= 0.5:
#         yp = 1;
#     else:
#         yp = 0
        
#     if yp == y_v[i]:
#         score = score + 1
    
# accuracy = score/len(y_v)
_, accuracy = model.evaluate(X_v, y_v)
print('Accuracy: %.2f' % (accuracy*100))