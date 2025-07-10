# -----------------------------------------------------------------------------
# Copyright (c) 2025 Blake Wilkey
#
# Submitted for Tako – Take-Home Coding Assessment (July 2025)
#
# This code is the original work of Blake Wilkey and is provided solely for the
# purpose of evaluation as part of the hiring process. All rights are reserved.
# -----------------------------------------------------------------------------

import os
import gc
import nltk
import json
import ollama
import time
import requests
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

from contextlib import contextmanager
from kneed import KneeLocator
from nltk.tokenize import sent_tokenize
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from tqdm import trange


def embed_sentences(text, transformer_name='all-MiniLM-L6-v2', transformer_dir='./model_cache/sentence_transformer', 
                    tokenizer_name='punkt', tokenizer_dir='./model_cache/tokenizers', use_pca=False, save_fig=None):
    """
    Tokenizes input text into sentences, encodes them into vector embeddings using a transformer model,
    and optionally applies PCA for dimensionality reduction and variance visualization.

    Parameters:
    - text (str): The raw input text to process.
    - transformer_name (str): Name of the SentenceTransformer model to use.
    - transformer_dir (str): Directory to cache the transformer model.
    - tokenizer_name (str): NLTK tokenizer to use (default: 'punkt').
    - tokenizer_dir (str): Directory to store/download NLTK tokenizer data.
    - use_pca (bool): Whether to apply PCA to reduce embedding dimensionality.
    - save_fig (str): Optional path to save a PCA variance plot.

    Returns:
    - np.ndarray: The resulting sentence embeddings (optionally PCA-reduced).

    """
    #Tokenize Input into well defined sentences/clauses
    nltk.data.path.append(tokenizer_dir)
    try:
        nltk.data.find(f'tokenizers/{tokenizer_name}')
    except LookupError:
        nltk.download(tokenizer_name, download_dir=tokenizer_dir)
        nltk.download('punkt_tab', download_dir=tokenizer_dir)
    sentences = np.array(sent_tokenize(text))

    #Transform sentences into vector embeddings.
    model = SentenceTransformer(transformer_name, 
                                cache_folder=transformer_dir)
    embeddings =  model.encode(sentences, 
                                normalize_embeddings=True)
    

    if use_pca:
        Pca = PCA(n_components=min(embeddings.shape[0], embeddings.shape[1]))
        Pca.fit(embeddings)
        explained_variance_ratio = Pca.explained_variance_ratio_
        embeddings = Pca.transform(embeddings)
        cumulative_explained_variance = explained_variance_ratio.cumsum()
        
        if save_fig:
            plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', label='Cumulative Variance')
            plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, label='Individual Variance')
            plt.title('PCA Explained Variance')
            plt.xlabel('Principal Component')
            plt.ylabel('Variance Explained')
            plt.xticks(range(1, len(explained_variance_ratio) + 1, 5))
            plt.legend()
            plt.grid(True)
            plt.savefig(save_fig)
            plt.clf()
    
    return sentences, embeddings


def find_optimal_clusters(embeddings:np.array, max_clusters:int, save_fig:str) -> np.array:
    """
    Fits multiple Gaussian Mixture Models (GMMs) to the given embeddings and selects the optimal 
    number of clusters using the elbow method (via KneeLocator) or gain percentage fallback.

    Parameters:
    - embeddings (np.ndarray): The input sentence embeddings of shape (n_samples, n_features).
    - max_clusters (int): The maximum number of clusters to evaluate (minimum is 2).
    - save_fig (str): Optional path to save the log-likelihood plot.

    Returns:
    - np.ndarray: Cluster assignments for each embedding using the optimal GMM.

    """
    log_likelihoods = np.zeros((10, max_clusters - 1))
    optimal_cluster_size = max_clusters

    for i in trange(0, 5, desc="Fitting EM Clusters"):
        for n in range(2, max_clusters + 1):
            gmm = GaussianMixture(n_components=n, reg_covar=1e-3)
            gmm.fit(embeddings)
            log_likelihoods[i][n-2] = gmm.lower_bound_
    log_likelihoods = log_likelihoods.mean(axis=0)
    plot_log_likelihoods(log_likelihoods, save_fig)

    #Optimal cluster computed with Maximum Curvature (Elbow Method) 
    knee = KneeLocator(list(range(2, len(log_likelihoods) + 2)), 
                                  log_likelihoods, curve='convex', 
                                  direction='increasing', interp_method='polynomial').knee
    
    #Curve not ideal for knee locator, use gain percentage
    if not knee or knee >= len(log_likelihoods):
        gains = np.diff(log_likelihoods)
        ratios = gains[1:] / gains[:-1]
        for i, r in enumerate(ratios, start=2 + 1):
            if r < .50:
                optimal_cluster_size = i
                break
    else:
        optimal_cluster_size = knee

    gmm = GaussianMixture(n_components=optimal_cluster_size, reg_covar=1e-3)
    gmm.fit(embeddings)
    return gmm.predict(embeddings)

def plot_log_likelihoods(log_likelihoods, save_fig):
    plt.plot(range(2, len(log_likelihoods) + 2), log_likelihoods, color='black')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Log Likelihood')
    plt.title('Expectation Maximization Log Likelihood')
    plt.grid(True)
    plt.savefig(save_fig)
    plt.clf()
    return  

def generate_prompt(prompt, model='qwen3:4b', model_dir='', think=False):
    if model_dir:
        os.environ['OLLAMA_MODELS'] = model_dir

    response = ollama.chat(model=model, 
                           messages=[{'role': 'user', 
                                      'content': prompt}],
                           think=think)
    
    return response['message']['content']

def generate_image(prompt, label, server_address='127.0.0.1:8188', workflow='./Comfyui/user/default/workflows/wf3.json', prompt_node_id='2', label_node_id='34', model='Flux-dev'):
    def _poll_comfy_prompt(prompt_id, endpoint='/history/', timeout=1200, interval=30):
        url = f"http://{server_address}{endpoint}{prompt_id}"
        start_time = time.time()
        while time.time() - start_time < timeout:

            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"Failed getting prompt from ComfyUI - HTTP {response.status_code}: {response.text}")

            data = response.json().get(prompt_id)
            if not data:
                time.sleep(interval)
                continue

            status = data.get("status", {})
            status_str = status.get("status_str", "unknown")

            if status_str == "error":
                raise Exception(f"Comfy UI server raised an exception - HTTP {response.status_code}: {response.text}")
            elif status.get("completed"):
                return 
            else:
                print(f"⏳ Status: {status_str}")
                time.sleep(interval)

        raise TimeoutError(f"⏰ Timed out waiting for prompt {prompt_id} to complete.")



    workflow_data = {}
    with open(workflow, 'r') as _workflow:
        workflow_data = json.load(_workflow)

    workflow_data[prompt_node_id]['inputs']['text'] = prompt
    workflow_data[label_node_id]['inputs']['filename_prefix'] = label

    result = requests.post(f'http://{server_address}/prompt',
                  json={'prompt': workflow_data})
    
    if result.status_code not in [200]:
        raise Exception(f'Failed Posting Prompt: "{prompt}" to Comfyui Server')
    
    prompt_id = result.json().get("prompt_id")

    _poll_comfy_prompt(prompt_id)

    return

def copy_files(target_dir, source_dir):
    for file in os.listdir(source_dir):
        if file.endswith(".png"):
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(target_dir, file)
            shutil.copy2(src_path, dst_path)
            os.remove(src_path)
    return


@contextmanager
def ollama_session():
    proc = start_ollama()
    try:
        yield
    finally:
        shutdown_process(proc)

@contextmanager
def comfyui_session():
    proc = start_comfyui()
    try:
        wait_response('http://127.0.0.1:8188/object_info')
        yield
    finally:
        shutdown_process(proc)

def wait_response(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError("Timed out waiting for ComfyUI to start")


def start_ollama():
    """
    Ensures the Ollama server is running. If not, starts it and waits until it's ready.

    Parameters:
    - host (str): Server address and port associated with the Ollama server. ie: 'http:127.0.0.1:11434'
    - timeout (int): Max seconds to wait for the server to become responsive.

    Returns:
        subprocess.Popen: The process handle for the ollama server.

    """

    #Start Ollama server
    return subprocess.Popen(['ollama', 'serve'], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)

def start_comfyui(comfyui_dir="./ComfyUI", port=8188, auto_launch=False):
    """
    Starts the ComfyUI server as a subprocess.

    Parameters:
        comfyui_dir (str): Path to the root of the ComfyUI installation.
        port (int): Port to run the server on (default is 8188).
        auto_launch (bool): Whether to auto-launch the browser UI.

    Returns:
        subprocess.Popen: The process handle for the ComfyUI server.
    """
    command = [
        "python3",
        "main.py",
    ]

    x = subprocess.Popen(
        command,
        cwd=comfyui_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return x

def shutdown_process(process):
    """
    Gracefully shuts down the ComfyUI subprocess.

    Parameters:
        process (subprocess.Popen): The process handle returned by start_comfyui().
    """
    if process:
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception as e:
            print(f"Warning: Failed to shut down ComfyUI cleanly: {e}")

