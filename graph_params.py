# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 18:52:14 2021

@author: paisa

graph_params.py
"""

def mean_neighbor_degree(Graph, node):
    mean_ND = 0;
    for n in Graph.neighbors(node):
        mean_ND = mean_ND + Graph.degree(n);
    
    mean_ND = mean_ND/(Graph.degree(node));
    
    return mean_ND;

def mean_smaller_neighbor_degree(Graph, node):
    mean_ND = 0;
    i = 0;
    for n in Graph.neighbors(node):
        if Graph.degree(n) <= Graph.degree(node):
            mean_ND = mean_ND + Graph.degree(n); 
            i = i+1;
    if i == 0:
        mean_ND = 0;
    else:
        mean_ND = mean_ND/i;
    
    return mean_ND;