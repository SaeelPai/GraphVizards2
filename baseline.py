import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
import pandas as pd
import os
from sklearn.metrics import mean_squared_error,r2_score
import datetime
import networkx as nx
from tqdm import tqdm

#Reading dummy train and val and removing nodes 1 and 2
cc_train=pd.read_csv("dummy_cctrain.csv")
cc_train = cc_train[(cc_train.src != "1") & (cc_train.src != "2")]
cc_train = cc_train[(cc_train.dst != "1") & (cc_train.dst != "2")]

cc_val=pd.read_csv("dummy_ccval.csv")
cc_val = cc_val[(cc_val.src != "1") & (cc_val.src != "2")]
cc_val = cc_val[(cc_val.dst != "1") & (cc_val.dst != "2")]


#Creating graph from both datasets
G = nx.from_pandas_edgelist(cc_train, source='src', target='dst')
G_val = nx.from_pandas_edgelist(cc_val, source='src', target='dst')

#Finding fully-connected subgraph from each of the two Graphs
sub_grps=[G.subgraph(c).copy() for c in nx.connected_components(G)]
G_main=sub_grps[0]

sub_grps_val=[G_val.subgraph(c).copy() for c in nx.connected_components(G_val)]
G_val=sub_grps_val[0]


total_conc_main=G_main.number_of_nodes()

##Reading in pagerank for each node csv and filtering out less important nodes
##Threshold chosen so as to ensure reasonable time on subsequent algo run
pr=pd.read_csv("pr_Gmain_dummy_cctrain.csv")
nodes_rmv=list(pr[pr["PageRank"]<=0.00005]["node"])

for node in nodes_rmv:
    
    G_main.remove_node(node)

##Finding concepts common to both training and val data
unique_conc_train=list(G_main.nodes())#list(set(cc_train[["src","dst"]].values.ravel("K")))
total_conc_train=len(unique_conc_train)

unique_conc_val=list(G_val.nodes())#list(set(cc_val[["src","dst"]].values.ravel("K")))
total_conc_val=len(unique_conc_val)

common_elements = np.intersect1d(unique_conc_train, unique_conc_val)

print(f"Total Concepts common to both datasets: {len(common_elements)}")



####Below code ran once to get adamic adar index for all link predictions in G_main####
####Obtained dataframe is saved for future use####
#pred_links=list(nx.adamic_adar_index(G_main))
#pl_df=pd.DataFrame(pred_links)

#Removing nodes in validation Graph that do not occur in training graph
nodes_train=list(G_main.nodes())
for node in unique_conc_val:
    
    if(node in nodes_train):
        continue
    else:
        G_val.remove_node(node)

#Reading in adamic_adar index csv and sorting it by index value
pl_df=pd.read_csv("Adamic_adar_pred.csv")
pl_df=pl_df.sort_values(by="2", ascending=False)

#Choosing number of predictions for evaluation equal to edges/connections in
#Validation graph G_val
pl_df_sub=pl_df.iloc[0:G_val.number_of_edges()]

#Storing connections in G_val in list
val_edges=list(G_val.edges())


#Iterating through all filtered predictions filtered in pl_df_sub and checking
#if they exist in val_edges. Took approx 5 minutes to run
tp=0
for idx,row in pl_df_sub.iterrows():
    
    if((row["0"],row["1"]) in val_edges or (row["1"],row["0"]) in val_edges):
        
        tp=tp+1
        
print(f"Positive prediction Accuracy is {tp/len(val_edges)}")




