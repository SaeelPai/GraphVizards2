import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
import pandas as pd
import os
from sklearn.metrics import mean_squared_error,r2_score
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from math import floor
import datetime
import networkx as nx
from operator import itemgetter
from graph_params import *
from scipy import stats



cc=pd.read_csv("dummy_data/dummy_cctrain.csv")
cc=cc.dropna()
cc=cc.drop_duplicates()


G = nx.from_pandas_edgelist(cc, source='src', target='dst') 
#nx.draw(g)

unique_conc=list(set(cc[["src","dst"]].values.ravel("K")))
total_conc=len(unique_conc)

print(f"Total concepts in edges_cc {len(unique_conc)}")
print("")
sub_grps=[G.subgraph(c).copy() for c in nx.connected_components(G)]

print(f"No. of connected subgraphs {len(sub_grps)}")
print("'")
for sub_grp in sub_grps:
    
    print(f"Total nodes in this sub graph are {sub_grp.number_of_nodes()}")

G_main=sub_grps[0];
G_main_nodes = list(G_main);
total_conc_main=G_main.number_of_nodes()
print("")
print(f"Total number of nodes removed after finding connected graph {total_conc-total_conc_main}")

#btwn_cent=nx.betweenness_centrality(G_main,10)

deg_cent_dict=nx.degree_centrality(G_main)   

deg_cent_val=np.fromiter(deg_cent_dict.values(),dtype=float)

#____________________________________________________________________________
## Main Graph stats and parameters

# degree histogram - list gives the frequency of each degree denoted by the index of the list plus 1
deg_histogram = nx.degree_histogram(G_main); 
del deg_histogram[0]; 
max_degree = len(deg_histogram);

# getting values for a linear fit of hte log log plot of degrees and frequency
#slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(range(1,len(deg_histogram)+1)),np.log(deg_histogram));

xx = np.linspace(1,len(deg_histogram),len(deg_histogram))

plt.figure(1)
plt.semilogx(xx,deg_histogram, c='b',linewidth=3)
plt.xlabel('Node Degree');
plt.ylabel('Number of Nodes');


nodes_by_deg = sorted(G_main.degree(),key=itemgetter(1),reverse=True);      #type is list
big_node = nodes_by_deg[0][0]                                           # node with highest degree

# finding the neighbours of big_node which have a degree of 1
big_node_1DN = [];
for n in G_main.neighbors(big_node):
    if G_main.degree(n)==1:
        big_node_1DN.append(n)

# calculating the mean neighbor degree of for nodes, starting from the high degree nodes
# calculating the mean neighbor degree considering only those nodes which have a degree
# lesser than the original node - something like downstream node av degree
mean_ND = [];
mean_smaller_ND = [];
for i in range(len(nodes_by_deg)):
    mean_ND.append(mean_neighbor_degree(G_main,nodes_by_deg[i][0]))
    mean_smaller_ND.append(mean_smaller_neighbor_degree(G_main,nodes_by_deg[i][0]))
    
plt.figure(2)
plt.plot(range(len(nodes_by_deg)),mean_smaller_ND);
plt.xlabel('nodes in order of descending degree');
plt.ylabel('average degree of neighbours with lesser degree')

# Page Rank default - stored as a dict
pr = nx.pagerank(G_main)

#_____________________________________________________________________________
## Cliques and related things
max_cliques_list = list(nx.find_cliques(G_main));
clique_number = nx.graph_clique_number(G_main, cliques=max_cliques_list)

clique_histogram = np.zeros(clique_number);
for i in range(len(max_cliques_list)):
    s = len(max_cliques_list[i]);
    clique_histogram[s-1] = clique_histogram[s-1] + 1;
    
plt.figure(3)
plt.plot(range(1,len(clique_histogram)+1),clique_histogram)
plt.xlabel('size of clique');
plt.ylabel('number of cliques')


      


