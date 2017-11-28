# MHP
This repository contains a python implementation for multivariate hawkes process.
It contains the following code components:

- ```HP/simulators```: To generate a cascade with given parameters using the modified Ogata's thinning algorithm.
- ```HP/estimators```: To estimate the parameters of multivariate hawkes process using maximum likelihood.

The simulation code is copied and modified slightly from [Steve Morse's excellent implementation](https://github.com/stmorse/hawkes). 
The code is highly optimized to get runtime improvements compared to naive implementations.
