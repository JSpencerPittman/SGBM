# Semi-Global Block Matching (SGBM)
This repository was implemented for CS 5802, *"Introduction to Parallel Programming and Algorithms"*.

To run this repository make sure you have `OpenCV` installed in a place that cmake will be able to find it. 
Then navigate to the `build` directory and run `cmake ..` and `make` to build the 
project from source code. This will produce three executables.
1. `SGBM`, my CUDA implementation of SGBM.
2. `OPENCV_BM`, OpenCV version of SGBM to act as a benchmark.
3. `EVAL`, used to evaluate a prediction in the RMSE and <3px metrics.

The `build/eval` script can be used to generate depth maps and evaluate them for the
sample pairs in `data`.
