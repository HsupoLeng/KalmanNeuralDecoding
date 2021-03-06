﻿# KalmanNeuralDecoding
This project recapitulates *Wu et al.*'s work on neural decoding in a sequential reaching task using a switching Kalman filter [\[1\]](https://www.ncbi.nlm.nih.gov/pubmed/15188861) and a linear Kalman filter [\[2\]](http://www.dam.brown.edu/people/elie/papers/Wu%20et%20al%20NIPS%2003.pdf). Decoding performance is characterized in terms of position prediction MSE. Performance improvement of the switching Kalman filter compared to the linear Kalman filter is comparable to that reported in the original paper. The dataset I use is prepared by *Perich et al.* [\[3\]](http://crcns.org/data-sets/motor-cortex/pmd-1/about-pmd-1), originally appearing in the paper by *Lawlor et al.* [\[4\]](https://link.springer.com/article/10.1007%2Fs10827-018-0696-6), and now published on CRCNS.org. 

## Running the code
Run `kalman_filter_demo` script in MATLAB to check the result. 

## Ongoing work
There is another nonlinear filter implemented in this repository, which uses a kinematic state model with a nonlinear transform function and a Poisson observation model. This filter is based on the work of *Yu et al.*[\[5\]](https://link.springer.com/chapter/10.1007/978-3-540-69158-7_61). It is NOT working currently, and is still under development. 

## Related material
For an overview of the project, see [my slides](https://drive.google.com/open?id=1vXbpNl9WGV5ywvMnR2quSIFxiqGjiB4b); for more details, refer to [my project report](https://drive.google.com/open?id=19JCDDboIuSYYy22qNv-5eyzc07g4EBKI). 
