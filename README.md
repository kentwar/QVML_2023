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
Aruane M. Pineda, Paul Kent, Caroline L. Alvesa, J ̈urgen Branke, Colm Connaughto, Francisco A. Rodrigues

### Brief outline of code
#### Purpose and disclaimer
This code produces the networks used in the above paper and runs the Q-voter model experiments, producing the data used in the paper.
While the random seed is not set in the code, which means the exact results produced in the paper will not be identically reproduced, the outcomes were
reproduced several times throughout the production of this work and the authors are confident that anyone running this code will achieve similar results.

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
