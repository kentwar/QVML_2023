# In[1]:
import networkx as nx
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import igraph
import numba
import numpy as np


# In[2]:


##TODO - Get the correct networks loaded.
experiments = ['1-Airtrafic', '2-Network_Science', '3-Political', '4-Euroroad','5-Crime',
               '6-Film_Trust', '7-hamster_household', '8-Blogs', '9-Yeast', '10-friendships-hamster', '11-Rovira']
dir = 'nets/Redes_Sociais/'
dirs = [dir]*len(experiments)
for i in range(1,len(experiments)):#len(experiments)):      
    exp = experiments[i]
    dir_ = dirs[i]
    print(exp)
    print(dir_)
    def media_valores(fun):     
        vk = dict(fun)
        vk = list(vk.values())
        vk = np.array(vk)
        md = np.mean(vk)
        return md

    def mean_dic(dic):
        return(
            sum(dic.values()) / len(dic) 
        )

    # In[3]:


    def Qvoter(GBA,matriz,q,p):
        soma=np.zeros(2)

        # [En] Difference from the original model
        ## In the original model, only one vertex is updated and in 
        # the current model, the N vertices of the network are updated
        nodes = list(GBA.nodes)

        for i in range(N):
            V1 = random.choice( nodes ) # choose a random vertex from the network
            prob = random.random()
            if prob < (1-p): #conformity
                v = [n for n in GBA.neighbors(V1)] # list all neighbors of V1

                v_sel = []
                if len(v)> 0:
                    for j in range(q): #select q neighbors within the list, with a chance of repetition
                        v_sel.append(random.choice(v))

                    for j in range(2): #possible states that can assume 0 or 1 : sum[0] and sum[1]
                        soma[j]=0

                    for j in v_sel: # visit all neighbors, count how many neighbors are 0s and how many are 1s
                        soma[int(matriz[j])] = soma[int(matriz[j])] +1     

                    if soma[0] == q:
                        matriz[V1] = 0
                    elif soma[1] == q:
                        matriz[V1] = 1
                    else: # if it tied (sum[0] = sum[1])
                        prob = random.random()
                        if prob < 0.2:
                            if matriz[V1] == 0:
                                matriz[V1] = 1
                            else :
                                matriz[V1] = 0
            else: #nonconformity (change independent of neighbors)
                prob = random.random()
                if prob < 0.5:
                    matriz[V1] = 0
                else:
                    matriz[V1] = 1


    # @numba.jit(nopython=True)
    # def Qvoter2(adj_matrix, matriz, q, p, N):
    #     sum_ = np.zeros(2)

    #     nodes = np.arange(N)
    #     for i in range(N):
    #         V1 = np.random.choice(nodes)  # choose a random vertex from the network
            
    #         prob = np.random.random()
    #         if prob < (1 - p):  # conformity
    #             v = np.nonzero(adj_matrix[V1])[0]  # list all neighbors of V1

    #             if len(v) > 0:
    #                 v_sel = np.random.choice(v, size=q, replace=True)  # select q neighbors within the list, with a chance of repetition

    #                 # possible states that can assume 0 or 1: sum[0] and sum[1]
    #                 sum_ = np.bincount(matriz[v_sel].astype(np.int64), minlength=2)

    #                 if sum_[0] == q:
    #                     matriz[V1] = 0
    #                 elif sum_[1] == q:
    #                     matriz[V1] = 1
    #                 else:  # if it tied (sum[0] = sum[1])
    #                     prob = np.random.random()
    #                     if prob < 0.2:
    #                         matriz[V1] = 1 - matriz[V1]
    #         else:  
    #             # nonconformity (change independent of neighbors)
    #             prob = np.random.random()
    #             matriz[V1] = int(prob >= 0.5)

    @numba.jit(nopython=True)
    def Qvoter2(adj_matrix, matriz, q, p, N):
        sum_ = np.zeros(2)

        nodes = np.arange(N)
        changes_count = 0  # Add a counter to track the changes

        for i in range(N):
            V1 = np.random.choice(nodes)  # choose a random vertex from the network

            prob = np.random.random()
            if prob < (1 - p):  # conformity
                v = np.nonzero(adj_matrix[V1])[0]  # list all neighbors of V1

                if len(v) > 0:
                    v_sel = np.random.choice(v, size=q, replace=True)  # select q neighbors within the list, with a chance of repetition

                    # possible states that can assume 0 or 1: sum[0] and sum[1]
                    sum_ = np.bincount(matriz[v_sel].astype(np.int64), minlength=2)

                    old_state = matriz[V1]  # Store the old state before updating

                    if sum_[0] == q:
                        matriz[V1] = 0
                    elif sum_[1] == q:
                        matriz[V1] = 1
                    else:  # if it tied (sum[0] = sum[1])
                        prob = np.random.random()
                        if prob < 0.2:
                            matriz[V1] = 1 - matriz[V1]

                    if old_state != matriz[V1]:  # If the state changed, increment the counter
                        changes_count += 1
            else:
                # nonconformity (change independent of neighbors)
                prob = np.random.random()
                old_state = matriz[V1]  # Store the old state before updating
                matriz[V1] = int(prob >= 0.5)

                if old_state != matriz[V1]:  # If the state changed, increment the counter
                    changes_count += 1

        return changes_count
    # In[4]:

    def nx_adjacency_to_numpy(graph):
        adj_matrix = nx.to_numpy_array(graph, dtype=int)
        return adj_matrix

    def networkx_to_igraph(nx_graph):
        ig_graph = igraph.Graph(len(nx_graph))
        ig_graph.vs["name"] = list(nx_graph.nodes())
        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes())}
        ig_graph.add_edges([(node_to_index[edge[0]], node_to_index[edge[1]]) for edge in nx_graph.edges()])
        return ig_graph

    tic = time.time()
    n_redes = 1
    lista_final =[]
    GBA= nx.read_edgelist(dir_ + exp + '.txt', nodetype=int)
    GBA = nx.relabel.convert_node_labels_to_integers(GBA, first_label=0, ordering='default')
    GBA = GBA.to_undirected() 
    connected_components = nx.connected_components(GBA)
    subgraphs = [GBA.subgraph(component) for component in connected_components]
    Gcc = sorted(subgraphs, key=len, reverse=True)
    GBAconnect=Gcc[0]
    ig_GBA = networkx_to_igraph(GBA)
    #igraph is faster for most of the calculations
    ig_clustering = ig_GBA.transitivity_local_undirected(mode="zero")
    nx_clustering = {node: c for node, c in zip(ig_GBA.vs["name"], ig_clustering)}
    C = mean_dic(nx_clustering)
    ig_closeness_centrality = ig_GBA.closeness(normalized=True)
    nx_closeness_centrality = {node: cc for node, cc in zip(ig_GBA.vs["name"], ig_closeness_centrality)}
    CLC = mean_dic(nx_closeness_centrality)
    BC  = media_valores(nx.betweenness_centrality(GBA))
    ig_eigenvector_centrality = ig_GBA.eigenvector_centrality(directed=False, scale = False)
    nx_eigenvector_centrality = {node: ec for node, ec in zip(ig_GBA.vs["name"], ig_eigenvector_centrality)}
    EC = mean_dic(nx_eigenvector_centrality)
    SPL = ig_GBA.average_path_length(directed=False)
    PC = ig_GBA.assortativity_degree(directed=False)
    IC= media_valores(nx.information_centrality(GBAconnect))
    SC=media_valores(nx.subgraph_centrality(GBA))
    AC= media_valores(nx.approximate_current_flow_betweenness_centrality(GBAconnect))
    ig_pagerank = ig_GBA.pagerank(directed=False, damping=0.85)
    nx_pagerank = {node: pr for node, pr in zip(ig_GBA.vs["name"], ig_pagerank)}
    PR = mean_dic(nx_pagerank)
    ig_core_number = ig_GBA.coreness()
    nx_core_number = {node: cn for node, cn in zip(ig_GBA.vs["name"], ig_core_number)}
    KC = mean_dic(nx_core_number)

    lista_final.append((C,CLC,BC,EC,SPL,PC,IC,SC,AC,PR,KC))
    print(f'Finding network info took: {time.time()-tic} seconds')
    df_caracterizadores = pd.DataFrame(lista_final,columns = ['C','CLC','BC','EC','SPL','PC','IC','SC','AC','PR','KC'])
    df_caracterizadores = df_caracterizadores.dropna() 
    #print(df_barabasi)
    # export_csv = df_barabasi.to_csv(r'Erdos_TM.csv', index = None, header=['C','CLC','BC','EC','SPL','PC','IC','SC','AC','PR','KC','tempo'])


    # In[5]:


    #N = 1000
    

    #Parametros do Qvoter
    pct_votante = 0.20
    q = 2
    p = 0.01

    #Parametros da simulação
    # n_sim = 50
    # n_redes = 50
    n_sim = 100

    import time

    import multiprocessing as mp
    GBA= nx.read_edgelist(dir_ + exp + '.txt', nodetype=int)
    GBA = nx.relabel.convert_node_labels_to_integers(GBA, first_label=0, ordering='default')
    N = len(GBA)
    adj_mat = nx_adjacency_to_numpy(GBA)
    media_tempo = []
    def simulate_qvoter(k):              
        tempo = 0
        matriz = np.zeros(N)
        seeds = random.sample(range(0, N), int(pct_votante*N))
        matriz[seeds] = 1
            # Get a list of (node, degree) tuples and sort by degree in descending order
            #degree_sorted_nodes = sorted(GBA.degree(), key=lambda x: x[1], reverse=False)

            # Get the top `int(pct_votante * N)` nodes
            #top_nodes = degree_sorted_nodes[:int(pct_votante * N)]

            # Update matriz with 1 for the indices corresponding to the top nodes
            #for node, _ in top_nodes:
            #    matriz[node] = 1
        soma_matriz = sum(matriz)
        total_changes = 0
        while (soma_matriz !=0) and (soma_matriz !=N):
            total_changes += Qvoter2(adj_mat,matriz,q,p,N)      
            soma_matriz = sum(matriz)
            tempo = tempo+1
            # if tempo % 100 == 0:
            #     print(f'Iteração {tempo}')
            if tempo > 2e+5:
                return((2e+5))
        #media_tempo = media_tempo/n_sim
        return ((total_changes, tempo))

    tic = time.time()


    from joblib import Parallel, delayed
    n_sims=  1
    lista_final = Parallel(n_jobs=1)(delayed(simulate_qvoter)(k) for k in range(n_sims))
    media_tempo = np.mean(lista_final)
    df_result = pd.DataFrame([media_tempo], columns=['media_tempo'])
    df_caracterizadores['media_tempo'] = df_result['media_tempo']
    print(f'Optimised time: {time.time() - tic:.2f} s')
    export_csv = df_caracterizadores.to_csv(f'{exp}_Random.csv', index = None)


