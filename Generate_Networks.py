#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import pandas as pd
import igraph
from igraph import Graph
from statistics import mean
from networkx.generators.community import LFR_benchmark_graph
import os

# In[3]:


n_redes=100    #number of network to generate


# In[5]:


# Complex network: Barabasi Linear 
n0=5 
N = 1000
for k1 in range(n_redes):
    g1 = Graph.Barabasi(N, n0, outpref=True, directed=False, power=1.0, zero_appeal=1, implementation="psumtree", start_from=None)
    G1= g1.to_networkx()
    os.makedirs("nets/barabasi/", exist_ok=True)
    fh1 = open(f"nets/barabasi/Barabasi_linear" + str(k1) + '.edgelist', "wb")
    nx.write_edgelist(G1, fh1)


# In[6]:


# Complex network: Barabasi non-Linear(0.5) 
n0=5 
N = 1000
for k2 in range(n_redes):
    g2 = Graph.Barabasi(N, n0, outpref=True, directed=False, power=0.5, zero_appeal=1, implementation="psumtree", start_from=None)
    G2= g2.to_networkx()
    os.makedirs("nets/barabasi_05_nl/", exist_ok=True)
    fh2 = open(f"nets/barabasi_05_nl/Barabasi05_nonlinear" + str(k2) + '.edgelist', "wb")
    nx.write_edgelist(G2, fh2)


# In[7]:


# Complex network: Barabasi non-Linear(1.5) 
n0=5 
N = 1000
for k3 in range(n_redes):
    g3 = Graph.Barabasi(N, n0, outpref=True, directed=False, power=1.5, zero_appeal=1, implementation="psumtree", start_from=None)
    G3= g3.to_networkx()
    os.makedirs("nets/barabasi_15_nl/", exist_ok=True)
    fh3 = open("nets/barabasi_15_nl/Barabasi15_nonlinear" + str(k3) + '.edgelist', "wb")
    nx.write_edgelist(G3, fh3)


# In[8]:


# Complex network: Waxman
n=1000
for k4 in range(n_redes):
    connected = False
    while not connected:
    	G4 = nx.waxman_graph(n, beta=0.12, alpha=0.1, L=None, domain=(0, 0, 1, 1), metric=None, seed=None)
    	connected = nx.is_connected(G4)
    	print(f'Was the generated network fully connected? : {connected}')

    os.makedirs("nets/Wax/", exist_ok=True)
    fh4 = open(f"nets/Wax/wax_linear{str(k6)}.edgelist", "wb")
    nx.write_edgelist(G4, fh4)


# In[9]:


# Complex network: Erdos
p=0.01
N=1000
for k5 in range(n_redes):
    G5 = nx.erdos_renyi_graph(N, p) 
    os.makedirs("nets/erdos/", exist_ok=True)
    fh5 = open("nets/erdos/Erdos" + str(k5) + '.edgelist', "wb")
    nx.write_edgelist(G5, fh5)


# In[11]:


# Complex network: LFR
n=1000
tau1=3
tau2=1.5
muu=0.1
for k6 in range(n_redes):
    G6=LFR_benchmark_graph(n, tau1, tau2, muu, average_degree=10, min_degree=None, max_degree=None, min_community=100, max_community=None, tol=1e-07, max_iters=500,seed=10)
    G6.remove_edges_from(nx.selfloop_edges(G6))
    os.makedirs("nets/lfr/", exist_ok=True)
    fh6 = open("nets/lfr/LFR" + str(k6) + '.edgelist', "wb")
    nx.write_edgelist(G6, fh6)
    

# Complex network: Linear
k_avg = 10
for j in range(n_redes):
    G = nx.path_graph(n)
    # Calculate the total number of edges needed to achieve the desired average degree
    m = int(k_avg * n / 2)
    
    # Add or remove edges to adjust the degree sequence
    if m > G.number_of_edges():
        # Add edges to increase the average degree
        num_edges_to_add = m - G.number_of_edges()
        nodes = list(G.nodes())
        for i in range(num_edges_to_add):
            # Choose two nodes at random and add an edge between them
            u, v = np.random.choice(nodes, size=2, replace=False)
            G.add_edge(u, v)
  
    else:
        # Remove edges to decrease the average degree
        num_edges_to_remove = G.number_of_edges() - m
        edges = list(G.edges())
        for i in range(num_edges_to_remove):
            # Choose an edge at random and remove it
            u, v = np.random.choice(edges)
            G.remove_edge(u, v)

    # Verify that the resulting graph has the desired average degree
    k_avg_actual = sum(dict(G.degree()).values()) / n
    print("Desired average degree: ", k_avg)
    print("Actual average degree: ", k_avg_actual)
    os.makedirs("nets/Linear/", exist_ok=True)
    fh1 = open("nets/Linear/Linear" + str(j) + '.edgelist', "wb")

    nx.write_edgelist(G, fh1)
    
# Complex network: Watts-Strogatz (Watts)
# The Watts-Strogatz graph is a small-world network model that is not strictly linear, but it has some linear properties.
n_redes=100
k1=10
p=0.01 
os.makedirs("nets/Watts/", exist_ok=True)
for k in range(n_redes):
    G1= nx.watts_strogatz_graph(1000, k1, p)

    fh = open("nets/Watts/Watts" + str(k) + '.edgelist', "wb")
    nx.write_edgelist(G1, fh)



# In[21]:

# Complex network: Regular (Not used in Paper)
n_redes=100
degree=10
os.makedirs("nets/Regular/", exist_ok=True)
for i in range(n_redes):
    G=nx.random_regular_graph(degree,1000)
    fh1 = open("nets/Regular/Regular" + str(i) + '.edgelist', "wb")
    nx.write_edgelist(G, fh1)



# In[16]:

