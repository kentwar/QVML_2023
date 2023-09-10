# QVML_2023
Code for the paper 'Machine learning for the prediction of q-voter model stationary state through complex network structures' Submitted 2023 - Journal of Statistical Mechanics.


# Abstract 
In this paper, we consider machine learning algorithms to accurately predict two variables associated to the $Q$-voter model in complex networks, i.e., (i) the consensus time and (ii) the frequency of opinion changes. Leveraging nine topological measures of the underlying networks, we verify that the clustering coefficient (C) and information centrality (IC) emerge as the most important predictors for these outcomes. Notably, the machine learning algorithms demonstrate accuracy across three distinct initialization methods of the $Q$-voter model, including random selection and the involvement of high- and low-degree agents with positive opinions. By unraveling the intricate interplay between network structure and dynamics, this research sheds light on the underlying mechanisms responsible for polarization effects and other dynamic patterns in social systems. Adopting a holistic approach that comprehends the complexity of network systems, this study offers insights into the intricate dynamics associated with polarization effects and paves the way for investigating the structure and dynamics of complex systems through modern methods of machine learning.

### Authors 
[Aruane M. Pineda](https://github.com/Aruane), [Paul Kent](https://github.com/kentwar), Colm Connaughton, Francisco A. Rodrigues

### Brief outline of code
#### Purpose and disclaimer
This repository contains the source code utilized for generating the network structures and conducting the Q-voter model experiments presented in the accompanying research paper. Please note that due to the absence of a fixed random seed, the specific numerical results may vary slightly from those reported in the paper. However, the authors have extensively verified and validated the code, ensuring that the obtained outcomes closely align with the findings discussed in the publication.

As such, we are confident that users running this code will achieve comparable results and we encourage users to engage with the code and appreciate any feedback or suggestions for further improvement.

#### Requirements
The code requires python > 3.8
further requirements can be installed by navigating to the project folder and running 
```python
pip install -r requirements.txt
```
#### Usage - Generating Networks
The exact networks used in the paper can be found in the /nets/ folder. New networks can be produced using the 'Generate_Networks.py' script from the command line.

```python
python Generate_Networks.py
```

#### Usage - Running experiments
We can then generate the convergence data from the three initial configurations by running with three different values of s.
s = 0 -> Low degree starting nodes
s = 1 -> High degree starting nodes
s = 2 -> Random degree starting nodes 

```python
python RunQvoter.py --s 0
```

#### Notes on Parallelisation
This code is highly parallelised using the joblibs package. We use a cpu count of 30, which may produce errors on a computer with less nodes. In this case we direct the user to line 277 of the 'RunQvoter.py' file where they can modify the 'njobs = 30' variable to work with their own resources.
