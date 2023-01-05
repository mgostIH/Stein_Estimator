# Neural Stein Estimator

A simple project that trains a neural network to test the [Stein's Paradox](https://www.youtube.com/watch?v=cUqoHQDinCM).

It uses Pytorch and trains a standard feed forward neural network on generated data, the parameters at the beginning of the file `main.py` can be freely changed to explore different training behavior.
The goal is reached whenever the network beats (in terms of average mean squared error) the standard estimator.

P independent gaussian random variables of different mean (distributed uniformly, but this shouldn't matter) and variance 1 are sampled once each, the goal is to estimate the vector of means from the samples. The standard approach is to just consider the samples as your guess, but strangely you can do better whenever P > 2. This network is trained to become a better estimator.

The network is a feed forward with ReLU trained with Adam, it can train quick enough on CPU, consider using the GPU if increasing batch size and network width.