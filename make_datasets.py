# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:01:24 2021

@author: paisa

The data is formed as arrays with 3 columns
The first 2 are the node numbers as per the dict by Haresh
The 3rd column is the presence or absence of link (1,0)
"""
import numpy as np
import pandas as pd

# #----------------------------------Train Data----------------------------------
# # make train, test, val datasets from rengaraj's u,v samples

# train_pos_u = np.load('dummy_data/train_pos_u.npy')
# train_pos_v = np.load('dummy_data/train_pos_v.npy')
# train_in_pos = np.stack((train_pos_u,train_pos_v),axis = -1)
# train_out_pos = np.ones((train_in_pos.shape[0],1))
# train_pos_in_out = np.append(train_in_pos, train_out_pos, axis=1)


# train_neg_u = np.load('dummy_data/train_neg_u.npy')
# train_neg_v = np.load('dummy_data/train_neg_v.npy')
# train_in_neg = np.stack((train_neg_u,train_neg_v),axis = -1)
# train_out_neg = np.zeros((train_in_neg.shape[0],1))
# train_neg_in_out = np.append(train_in_neg, train_out_neg, axis=1)

# # the last column is whether or not the link is present. 1 is link present, 0 is link absent
# train_data = np.concatenate((train_pos_in_out,train_neg_in_out))
# np.random.shuffle(train_data)
# np.save('dummy_data/train_data',train_data)

# #----------------------------------Test Data-----------------------------------

# test_pos_u = np.load('dummy_data/test_pos_u.npy')
# test_pos_v = np.squeeze(np.load('dummy_data/test_pos_v.npy'))
# test_in_pos = np.stack((test_pos_u,test_pos_v),axis = -1)
# test_out_pos = np.ones((test_in_pos.shape[0],1))
# test_pos_in_out = np.append(test_in_pos, test_out_pos, axis=1)


# test_neg_u = np.load('dummy_data/test_neg_u.npy')
# test_neg_v = np.load('dummy_data/test_neg_v.npy')
# test_in_neg = np.stack((test_neg_u,test_neg_v),axis = -1)
# test_out_neg = np.zeros((test_in_neg.shape[0],1))
# test_neg_in_out = np.append(test_in_neg, test_out_neg, axis=1)

# # the last column is whether or not the link is present. 1 is link present, 0 is link absent
# test_data = np.concatenate((test_pos_in_out,test_neg_in_out))
# np.random.shuffle(test_data)
# np.save('dummy_data/test_data',test_data)
#----------------------------------Val Data------------------------------------

# val_csv = pd.read_csv('dummy_data/cc_val_wts.csv')
# val_char = val_csv[['src','dst']].values
# node_num2str=pd.read_csv('dummy_data/node_dict.csv')
# dict_num2str=dict(zip(node_num2str.nodes, node_num2str.int_names))
# val_data = []

val_pos_u = np.load('dummy_data/val_pos_u.npy')
val_pos_v = np.load('dummy_data/val_pos_v.npy')
val_in_pos = np.stack((val_pos_u,val_pos_v),axis = -1)
val_out_pos = np.ones((val_in_pos.shape[0],1))
val_pos_in_out = np.append(val_in_pos, val_out_pos, axis=1)


val_neg_u = np.load('dummy_data/val_neg_u.npy')
val_neg_v = np.load('dummy_data/val_neg_v.npy')
val_in_neg = np.stack((val_neg_u,val_neg_v),axis = -1)
val_out_neg = np.zeros((val_in_neg.shape[0],1))
val_neg_in_out = np.append(val_in_neg, val_out_neg, axis=1)

# the last column is whether or not the link is present. 1 is link present, 0 is link absent
val_data = np.concatenate((val_pos_in_out,val_neg_in_out))
np.random.shuffle(val_data)
np.save('dummy_data/val_data',val_data)

# # val_data only consist of links whose nodes are present in the dict which is
# # made on the basis on the nodes in train 
# for i in range(val_data_full.shape[0]):
#     if val_char[i,0] in dict_num2str:
#         if val_char[i,1] in dict_num2str:
#             val_data.append((dict_num2str[val_char[i,0]], dict_num2str[val_char[i,1]],1.))

# val_data = np.array(val_data)   
# np.random.shuffle(val_data)
# np.save('dummy_data/val_data',val_data)




