#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Copyright (c) 2025 Blake Wilkey
#
# Submitted for Tako – Take-Home Coding Assessment (July 2025)
#
# This code is the original work of Blake Wilkey and is provided solely for the
# purpose of evaluation as part of the hiring process. All rights are reserved.
# -----------------------------------------------------------------------------

import argparse
import os

from pdb import set_trace
from pipeline import Visual_asset_pipeline
from lib.utils import comfyui_session, ollama_session, copy_files

def execute_pipeline(filepath:str, client:str, random_clusters:bool) -> tuple:
    try:
        with open(filepath, 'r', encoding='utf-8') as meeting_notes:
            kwargs = {'notes': meeting_notes.read(), "client_id": client}
        
        #
        Pipeline = Visual_asset_pipeline(**kwargs)
        Pipeline.generate_sentence_embeddings()
        Pipeline.compute_optimal_clusters()
        
        with ollama_session():
            Pipeline.generate_image_prompts(random_clusters=random_clusters)
        
        with comfyui_session():
            Pipeline.generate_images()       
        
        copy_files('./out','./ComfyUI/output')
    except Exception as e:
        #TODO: Catch individual exceptions and log accordingly
        print(e)
    
    return 0, '' 

class Arg_parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Script that requires a file path input."
        )
        self._add_arguments()

    def _add_arguments(self):
        
        self.parser.add_argument('filepath',
            type=self._valid_file,
            help="Path to the input file (must exist).")
        self.parser.add_argument('--client', 
                                 help='Specifies a client name. Used for output filenames')
        self.parser.add_argument('--random-clusters', 
                                 default=False, 
                                 help='Specifies whether to generate additional images from randomly sampled clusters')

    def _valid_file(self, path):
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError(f"File '{path}' does not exist.")
        return path

    def parse_args(self):
        return self.parser.parse_args()

if __name__ == "__main__":
    parser = Arg_parser()
    arg_info = vars(parser.parse_args())
    positional_keys = ["filepath"]
    args = [arg_info[k] for k in positional_keys]
    kwargs = {k: v for k, v in arg_info.items() if k not in positional_keys}
    execute_pipeline(*args, **kwargs)
