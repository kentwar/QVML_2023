# QVML_2023
Code for the paper 'Machine learning for the prediction of q-voter model stationary state through complex network structures' Submitted 2023 - Journal of Computational Science



# Abstract 
In network science, using structural features of systems to infer outcomes is crucial to understanding complex networks and their behaviors. Recently,
researchers have suggested a machine-learning-based approach for interpreting dynamic patterns in complex network systems. In this study, such tech-
niques are employed to estimate two outcomes, the time needed for reaching a stable state and the frequency of alterations in various complex networks
within the q-voter model, by leveraging the topological characteristics of these networks. Analysis was conducted on a random configuration of the
q-voter model and extended to two additional initial configurations, starting with high and low-degree nodes. This study reports excellent performance
in predicting both outcomes, provides a breakdown of the key network topological features used for estimation and ranks network metrics in order of
importance with high accuracy. This generalized approach applies to dynamical processes running on complex networks, representing a step towards
applying machine learning methods to studies of dynamical patterns in complex network systems.

### Authors 
Aruane M. Pineda, Paul Kent, Caroline L. Alves, J ̈urgen Branke, Colm Connaughton, Francisco A. Rodrigues

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
This code is highly parallelised using the joblibs package. We use a cpu count of 30, which may produce errors on a computer with less nodes. In this case we direct the user to line 277 of the 'RunQvoter.py' file where they can modify the code to work with their own resources.
