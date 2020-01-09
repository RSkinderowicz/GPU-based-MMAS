# GPU-based MAX-MIN Ant System

This is an implementation of the GPU-based MAX-MIN Ant System as described in 
*Skinderowicz, Rafał. "Implementing a GPU-based parallel MAX-MIN Ant System" Future Generation Computer Systems.*


## Preliminaries

CUDA 9.0 or newer is recommended. You also need a CUDA-enabled GPU.


## Building

Makefile should be adjusted so that the executable for the target GPU
architecture is created. For this purpose COMPUTE_CAPABILITY variable should
be set, for example:

    COMPUTE_CAPABILITY=$(CC_MAXWELL)

where CC_MAXWELL refers to the Maxwell architecture of the Nvidia GPUs

This software is intended to be compiled and run on Linux. 
It was tested with GCC v6.3 and CUDA 9.0.

To compile & build run:

    make

It could take a while. If everything goes OK, **mmas** executable should be
created.

## Running

The program takes a number of command line parameters, most of them should be
visible after running:

    ./mmas --help

An example execution of the MMAS-WRS-BT alg. could be started with:

    ./mmas --instance=ALL_tsp/pr1002.tsp --alg=mmas_cl_rs_bitmask

During the execution the program prints various diagnostic messages to the std.
output. The **final** results are stored in JSON file created in the directory
specified by **--out** parameter. By default they will be stored in the 
results/ folder.

Sample output obtained by executing the program on GeForce 840M mobile GPU:

    ./mmas --instance=ALL_tsp/pr1002.tsp --alg=mmas_wrs_ct_cl

    Read line: NAME : pr1002
    Read line: COMMENT : 1002-city problem (Padberg/Rinaldi)
    Read line: TYPE : TSP
    Read line: DIMENSION : 1002
    Read line: EDGE_WEIGHT_TYPE : EUC_2D
    Read line: NODE_COORD_SECTION
    Finished loading instance

    Starting trial 0

    NN sol. cost: 319540
    Max active #blocks for ant sol. build kernel: 28
    0:      Global | Reset | Iter. best costs: 278929 (7.67589%)    278929  278929
    100:    Global | Reset | Iter. best costs: 266898 (3.03152%)    266898  268875
    200:    Global | Reset | Iter. best costs: 261788 (1.05889%)    261788  261788
    300:    Global | Reset | Iter. best costs: 260356 (0.50609%)    260356  260356
    400:    Global | Reset | Iter. best costs: 260272 (0.473663%)   260272  260272
    500:    Global | Reset | Iter. best costs: 260238 (0.460538%)   260238  260238
    600:    Global | Reset | Iter. best costs: 260238 (0.460538%)   260238  260238
    700:    Global | Reset | Iter. best costs: 260179 (0.437762%)   260179  260179
    800:    Global | Reset | Iter. best costs: 260179 (0.437762%)   260179  260179
    900:    Global | Reset | Iter. best costs: 260107 (0.409967%)   260107  260107
    Elapsed (sec):7.55171
    Final solution cost: 260107 (0.409967%)
    Build sol. kernel mean time [ms]: 2.42799
    Evaporate pheromone mean time [ms]: 0.572934
    Heur. & pher. product cache mean update time [ms]: 0.914642
    Cand list product cache mean update time [ms]: 0.0562322
    Update best sol. mean time [ms]: 0.2047
    Deposit pheromone mean time [ms]: 0.020416
    Local search time [ms]: 3.23198
    Total time [s]: 7.42889
    Mean iter. time [ms]: 7.42889
    Saving results to: results/mmas_wrs_ct_cl-pr1002_2020-1-8_11_54_10.json

### Instance size vs the GPU memory size
The maximum size of the problem that can be solved depends on the amount of
the GPU-memory available. The MMAS implementation uses the "standard" approach
with a full pheromone matrix of size *n<sup>2</sup>*. 
Also several other matrices are stored in the
memory, i.e. with the distances between the nodes of the TSP instance solved,
along with the "heuristic knowledge" matrix and product cache matrix that stores 
the products of the pheromone and heuristic knowledge matrices.
For example, solving *pla7397* instance from TSPLIB repository takes approx.
900MB of the GPU RAM.