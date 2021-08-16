# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 21:49:09 2021

@author: 17657
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

node_num2str=pd.read_csv('dummy_data/node_dict.csv')
dict_num2str=dict(zip(node_num2str.int_names, node_num2str.nodes))
pr=pd.read_csv("node_metrics/pr_G_cc_train_wts.csv")
dict_pr=dict(zip(pr.Node, pr.PR))

#Change embedding name here
embedding_name="sdne"

#Read in the positve prediction file uploaded for Graphia
pos_pred=np.loadtxt(f"draft/for_graphia/best_pred_{embedding_name}.txt")    



def get_connection_rank(pos_pred,dict_num2str,dict_pr):
    
    
    link_imp=np.zeros((pos_pred.shape[0],1))
    
    for i,pred in enumerate(pos_pred):
        
        #pr_u=pr[pr['Node']==dict_num2str[int(pred[0])]]['PR'].values
        pr_u=dict_pr[dict_num2str[int(pred[0])]]
        pr_v=dict_pr[dict_num2str[int(pred[1])]]#pr[pr['Node']==dict_num2str[int(pred[1])]]['PR'].values
        #print(i)
        link_imp[i]=pr_u*pr_v
        
    pos_pred_imp=np.concatenate((pos_pred,link_imp),axis=1)
    
    return pos_pred_imp
    

pos_pred_imp=get_connection_rank(pos_pred,dict_num2str,dict_pr) 

df_imp=pd.DataFrame(pos_pred_imp)
df_imp.columns=["u","v","truth","pred","imp"]

df_imp=df_imp.sort_values(by='imp', ascending=False)

df_imp["color"]=""
df_imp.loc[df_imp["truth"]==1.0,"color"]="green"
df_imp.loc[df_imp["truth"]==0,"color"]="red"

df_imp["Index"]=np.arange(0,df_imp.shape[0],1)

########## Bar plot ##################

plt.figure(figsize=(8,6));
plt.bar([0,1],df_imp.groupby("truth")["imp"].mean());
plt.xticks([0,1],["False positives","True positives"]);
plt.ylabel("Average link importance");
plt.title(f"Average importance of predicted links: {embedding_name.upper()} embedding")



########## Box plot ##################
medianprops = dict(color="black",linewidth=2)

fig, ax = plt.subplots(figsize=(8,6))
df_imp.boxplot(column=['imp'], by='truth', ax=ax,showfliers=False,patch_artist=True,notch=True,medianprops=medianprops);
plt.grid(False);plt.xlabel("Ground truth");plt.xticks([1,2],["False positives","True positives"]);
plt.ylabel("Average link importance")
plt.title(f"Average importance of predicted links: {embedding_name.upper()} embedding");
plt.suptitle('')





     