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
from graph_params import *
from scipy import stats
from collections import OrderedDict
from scipy.stats import spearmanr

cc=pd.read_csv("dummy_cctrain.csv")
cc=cc.dropna()
cc=cc.drop_duplicates()

G = nx.from_pandas_edgelist(cc, source='src', target='dst') 
#nx.draw(g)

unique_conc=list(set(cc[["src","dst"]].values.ravel("K")))
total_conc=len(unique_conc)

sub_grps=[G.subgraph(c).copy() for c in nx.connected_components(G)]
G_main=sub_grps[0];
G_main_nodes = list(G_main);
total_conc_main=G_main.number_of_nodes()

#btwn_cent=nx.betweenness_centrality(G_main,10)

deg_cent_dict=nx.degree_centrality(G_main)   

deg_cent_val=np.fromiter(deg_cent_dict.values(),dtype=float)

# Page Rank default - stored as a dict and then a sorted dict
pr = nx.pagerank(G_main);
pr_sorted = OrderedDict(sorted(pr.items()));
pr_vals = list(pr_sorted.values());

# degree - stored as a dict and then a sorted dict
degree = {node:val for (node, val) in G_main.degree()};
deg_sorted = OrderedDict(sorted(degree.items()));
d_vals = list(deg_sorted.values());

#%% plotting
plt.figure(1)
plt.scatter(d_vals,pr_vals,c="r");
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('PageRank')
#plt.title('Degree vs PageRank Correlation')

#%% calculate spearman's correlation
coef, p = spearmanr(d_vals, pr_vals)
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
	print('Samples are uncorrelated. p=%.3f' % p)
else:
	print('Samples are correlated, p=%.3f' % p)







