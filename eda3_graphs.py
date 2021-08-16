# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 08:53:16 2021

@author: paisa
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
import pandas as pd
import os
from math import floor
import datetime
import networkx as nx
from operator import itemgetter
from scipy import stats
from collections import OrderedDict
from scipy.stats import spearmanr
from itertools import combinations

#%%

cc=pd.read_csv("dummy_data/cc_train_wts.csv")


G = nx.from_pandas_edgelist(cc, source='src', target='dst',edge_attr='count') 

unique_conc=list(set(cc[["src","dst"]].values.ravel("K")))
total_conc=len(unique_conc)


total_conc=G.number_of_nodes()

#btwn_cent=nx.betweenness_centrality(G_main,10)


# Page Rank default - stored as a dict and then a sorted dict
pr = nx.pagerank(G);
pr_sorted = OrderedDict(sorted(pr.items()));
pr_vals = list(pr_sorted.values());

# degree - stored as a dict and then a sorted dict
degree = {node:val for (node, val) in G.degree()};
deg_sorted = OrderedDict(sorted(degree.items()));
d_vals = list(deg_sorted.values());

#%% plotting
plt.figure(1)
plt.scatter(d_vals,pr_vals,c="r");
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('PageRank')
plt.title('Degree vs PageRank Correlation')

#%% calculate spearman's correlation
coef, p = spearmanr(d_vals, pr_vals)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('Samples are uncorrelated. p=%.3f' % p)
else:
	print('Samples are correlated, p=%.3f' % p)

#%%
cc_high_pr_list = [];
pr_cutoff = 1e-5;

for i in range(np.shape(cc)[0]):
    if pr[cc.iloc[i,0]] > pr_cutoff and pr[cc.iloc[i,1]] > pr_cutoff:
        cc_high_pr_list.append((cc.iloc[i,0], cc.iloc[i,1],cc.iloc[i,2],cc.iloc[i,3],cc.iloc[i,4]))
        
cc_high_pr_df = pd.DataFrame(cc_high_pr_list, columns=['src','dst','count','month','year'])
G_high_pr = nx.from_pandas_edgelist(cc_high_pr_df, source='src', target='dst',edge_attr='count')

sub_grps=[G_high_pr.subgraph(c).copy() for c in nx.connected_components(G_high_pr)]
G_main_hpr=sub_grps[0]

ave_pl = nx.average_shortest_path_length(G_main_hpr)
print('Average path length of the high PR graph = %.4f' % ave_pl)
length = dict(nx.all_pairs_shortest_path_length(G_main_hpr))
high_pr_nodes = list(G_main_hpr.nodes)
comb = list(combinations(high_pr_nodes, 2))

# adamic adar index - run this

