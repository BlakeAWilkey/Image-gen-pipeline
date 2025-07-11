
## TAKO Takehome Assessment: Blake Wilkey
## PATH 2 - Track 4: Visual Creation from Informal Briefings

## Prerequisites
Before running the script, ensure you have the following:

1. Linux environment
2. Conda package manager
3. Hugging Face Account
4. Nvidia GPU ideally with 10gb VRAM
5. 32gb-64gb system RAM 
6. 50gb storage


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


4. **Download Flux-dev Checkpoint file**

   The Checkpoint file can be downloaded from [hugging face](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors)

   NOTE: Do not modify the filenames

   After downloading, move the file into:
   ```bash
   ComfyUI/models/checkpoints
   ```

5. **Download Flux Text Encoders**
   clip_l can be downloaded from [hugging face](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors).

   t5_xxl encoder can be downloaded from [hugging face](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp16.safetensors)
   
   NOTE: Do not modify the filenames

   After both have finished downloading, move them into:
   ```bash 
   ComfyUI/models/clip
   ```
6. **Download VAE**

   Flux's vae can be downloaded from [hugging face](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main/vae)

   NOTE: Do not modify the filenames

   After the vae file has finished downloading, move it into:
   ```bash
   ComfyUI/models/vae
   ```

## Viewing Arguments
Once the environment is activated, you can view required and optional arguments for the launch script using:
```bash
./main.py -h 
```

## Executing the Pipeline
In order to generate images via the pipeline, you will need to execute 'main.py' and pass in a text file containing the meeting notes/informal briefing
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


## Viewing Plotted Graphs and System Design Overview

Graphs are saved in the 'graphs' directory. These include clustering log-likelihood and sentence-embedding PCA Eigenvalue graphs

An overview of the Pipeline's System Design is present in 'Pipeline_System_Design.png'
