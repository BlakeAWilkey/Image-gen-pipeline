## -----------------------------------------------------------------------------
## Copyright (c) 2025 Blake Wilkey
##
## Submitted for Tako – Take-Home Coding Assessment (July 2025)
##
## This project is the original work of Blake Wilkey and is provided solely for the
## purpose of evaluation as part of the hiring process. All rights are reserved.
## -----------------------------------------------------------------------------

## TAKO Takehome Assessment
## PATH 2, Track 4: Visual Creation from Informal Briefings
## Blake Wilkey


## Prerequisites
Before running the script, ensure you have the following:

1. Linux environment
2. Conda package manager
3. Nvidia GPU ideally with 10gb VRAM
4. At least 32gb of system RAM 


## Setup Instructions
1. **Install Conda**:
   - If Conda is not already installed, you can download and install it from the [official website](https://docs.conda.io/en/latest/miniconda.html).

2. **Create a Conda Environment**:
   - Make sure you are in the project directory (IMAGE_GEN_PIPELINE) containing the `environment.yaml` file.
   - Create the environment using the following command:
     ```bash
     conda env create -f environment.yaml -n bw
     ```

3. **Activate the Environment**:
   - Activate the Conda environment with:
     ```bash
     conda activate bw
     ```


## Viewing Arguments
Once the environment is activated, you can view required and optional arguments for the launch script using:
```bash
./main.py -h 
```

## Running an Experiment
In order to generate Images via the pipeline you will need to pass in a text file containing the meeting notes/informal briefing
NOTE: There are some examples in the data directory
```bash
./main.py data/Surecell_meeting_notes.txt
```


It is highly recommended that users specify a client id with the '--client' kwarg. This ensures that output images and graphs can be traced back to individual clients
```bash
./main.py data/Surecell_meeting_notes.txt --client Surecell
```

The pipeline utilizes unsupervised clustering methods. The '--random-clusters' kwarg will allow the pipeline to generate 3 additional images
by randomly sampling from clusters.

NOTE: On less robust systems this will require longer execution time. 
```bash
./main.py data/Surecell_meeting_notes.txt --client Surecell --random-clusters
```


## Viewing plotted graphs
Graphs are saved in the 'graphs' directory. These include clustering log-likelihood and sentence-embedding PCA Eigenvalue graphs
