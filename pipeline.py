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
import ollama
import random

import numpy as np
import matplotlib.pyplot as plt
import lib.utils as u

from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from tqdm import trange,tqdm

from pdb import set_trace

class Visual_asset_pipeline(object):
    def __init__(self, **kwargs):
        self.initial_notes = kwargs.get('notes', '')
        self.client_id = kwargs.get('client_id', '')
        self.embedding_model = kwargs.get('embedding_model', 'all-MiniLM-L6-v2')
        self.llm_model = kwargs.get('llm_model', 'qwen3:4b')
        self.image_checkpoint = "Flux1-dev"
        self.embeddings = np.array([])
        self.base_image_prompts = []
    
    def generate_sentence_embeddings(self):
        """
        Generates sentence embeddings from the initial client notes.

        This method:
        - Extracts individual sentences from the initial notes.
        - Computes dense vector embeddings for each sentence.
        - Optionally applies PCA for dimensionality reduction and visualization.
        - Stores both the raw sentences and their corresponding embeddings.
        - Saves a PCA plot of the principal components for inspection.

        Returns:
            None
        """

        self.sentences, self.embeddings = u.embed_sentences(self.initial_notes, 
                                            use_pca=True, 
                                            save_fig=f'./graphs/embedding_principal_components{self.client_id}')
        return

    def compute_optimal_clusters(self):
        """
        Computes the optimal number of clusters for the current sentence embeddings.

        This method:
        - Determines the optimal number of clusters using a model selection strategy (e.g., log-likelihood or BIC).
        - Stores the resulting cluster assignments.
        - Logs clustering diagnostics to a file for visualization and analysis.

        Returns:
            None
        """

        if self.embeddings.size == 0:
            return
        
        max_clusters = len(self.embeddings)
        self.cluster_assignments = u.find_optimal_clusters(self.embeddings, 
                                                           max_clusters, 
                                                           f'./graphs/clustering_log_likelihoods_client{self.client_id}')
        
        
    def generate_image_prompts(self, model_dir='./model_cache/ollama', num_variations=3, random_clusters=False):
        """
        Generates refined image prompts based on clustered client input.

        This method performs the following steps:
        1. Extracts a high-level visual deliverable description from initial notes via LLM.
        2. Summarizes each cluster of semantically grouped sentences into a descriptive phrase via LLM.
        3. Constructs a base image prompt using all clusters.
        4. Generates additional prompt variants by randomly sampling subsets of clusters.
        5. Refines each prompt into a Flux-optimized image prompt using a final LLM pass.
        6. Stores the final prompts with unique labels for downstream image generation.

        Parameters:
            model_dir (str): directory holding ollama models
            num_variations (int): Number of additional prompt variants to generate by sampling cluster combinations. (defaults to 3)
            random_clusters (bool): Generate additional image prompts by randomly sampling clusters

        Returns:
            None
        """
        image_prompts = []
        base_image_prompt = ''
        topic_result = ''
        cluster_phrases = {}
        
        if self.sentences.size == 0:
            return
        elif self.cluster_assignments.size == 0:
            return
        elif self.cluster_assignments.shape[0] != self.sentences.shape[0]:
            return

        #Initial Prompt attempts to extract the deliverable type, who the client is, and what they do
        topic_prompt = f'''What graphical deliverable is being described in the following text, 
                           for whom is it for, and what the goal of [entity] beyond this deliverable might be (Respond by filling in the brackets with necessary words 
                           "A [thing] for [entity] who [goal].") do not perform other formatting: "{self.initial_notes}"'''
        
        #Use Few-shot prompting to guide LLM Response for descriptive phrases
        descriptor_preamble = '''
                            Summarize the following sentences into one clear, well-formed descriptive phrase suitable for generating a logo concept. The phrase should read like a single sentence, not a list of keywords 
                            (only return the descriptive phrase):

                            Example:
                            Sentences:
                            - The client wants something clean and modern.
                            - They mentioned minimalism and simplicity.
                            - Avoid flashy colors or complex shapes.

                            -> Descriptive phrase: clean, minimalist design with simple shapes and a muted color palette


                            Now try:
                            Sentences:
                            '''
        
        #Refine each base image prompt to be optimized for image generation
        refine_prompt = '''Rewrite the following description into a single, concise optimized image prompt 
                           Focus on visual clarity, remove redundancy, and use concrete design elements like layout, color, shape, and style (only respond with the prompt):\n
                        '''

        #First sentence is the main goal
        with tqdm(total=1, desc='Determining Graphic Deliverable') as progress:
            topic_result = u.generate_prompt(topic_prompt, 
                              model=self.llm_model, 
                              model_dir=model_dir)
            topic = topic_result
            base_image_prompt += topic
            progress.update(1)
        
        base_image_prompt += ' '
        
        #Use all clusters to get one prompt
        for cluster_num in trange(0, np.max(self.cluster_assignments) + 1, desc="Generating Image Descriptors"):
            sentence_indices = np.where(self.cluster_assignments == cluster_num)[0]
            cluster_sentences = self.sentences[sentence_indices]
            formatted_descriptions = '\n'.join(f"- {s}" for s in cluster_sentences)
            condensed_phrase = u.generate_prompt(descriptor_preamble + formatted_descriptions, 
                                                 model=self.llm_model, 
                                                 model_dir=model_dir) + ' '
            cluster_phrases[cluster_num] = condensed_phrase
            base_image_prompt += condensed_phrase
        
        image_prompts.append((f'client{self.client_id}-{datetime.now().strftime("%Y%m%d_%H%M%S")}-all', base_image_prompt))

        #Exit if not generating additional images from randomly sampled clusters
        if random_clusters:
            #Randomly sample clusters results for additional prompts
            for i in trange(0, num_variations, desc="Generating Prompt Variants"):
                keys = list(cluster_phrases.keys())
                half = len(keys) // 2
                selected = random.sample(keys, half)
                variant = topic + ' ' + ' '.join(cluster_phrases[s] for s in selected)
                image_prompts.append((f'client{self.client_id}-{datetime.now().strftime("%Y%m%d_%H%M%S")}-{"".join(str(selected))}', variant))
            

        #Finally refine each prompt, save it within the pipeline object
        for label, prompt in tqdm(image_prompts, desc="Refining Image Prompts"):
            revised_prompt = u.generate_prompt(refine_prompt + prompt, 
                                               model=self.llm_model,
                                               model_dir=model_dir)    
            self.add_image_prompt(label, revised_prompt) 
        
    def add_image_prompt(self, label, prompt):
        self.base_image_prompts.append((label, prompt))
        
    def generate_images(self):
        """
        Generates images from finalized image prompts using the configured image generation backend.

        Iterates over all base image prompts and generates corresponding images,
        saving them to the specified output directory.

        Returns:
            None
        """

        for label, prompt in tqdm(self.base_image_prompts, desc="Generating Images"):
            u.generate_image(prompt, label)