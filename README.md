# Iterative Linear Quadratic Regulator

This is a parallelized, modern, extentable, single-header, C++11
implementation of the iterative Linear Quadratic Regulator (iLQR)
algorithm developed by Todorov, Tassa and Li.

This implementation is built in reference to the 2012 IROS paper
"Synthesis and Stabilization of Complex Behaviors through
Online Trajectory Optimization"
[link](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf).

An official matlab implementation of iLQR is available
[here](https://de.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization).

## Implementation details

All necessary derivatives are calculated with finite differences to
allow quick experimentation with the cost or dynamics function. 

Calculations are (as far as possible) parallelized using **openmp**.

Additionally the horizon length can be adjusted dynamically.

## Disclaimer

This is research code and does not adhere to industry standard
safety procedures or any extensive code checks.
This iLQR implementation **must** not be used in production. 
All rights for the algorithm remain with the respective inventors.
