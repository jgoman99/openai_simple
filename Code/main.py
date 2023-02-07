# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 23:38:28 2023

@author: Seldon
"""

import openai
from time import sleep
from numpy import dot
from numpy.linalg import norm
from pathlib import Path
import json
import numpy as np
import pandas as pd

# change this later, but lazy right now
# OpenAI API Key
with open("../Keys/openai_key.txt","r") as f:
    openai.api_key = f.read()

# default vars
default_embedding_cache_path = "embeddings_cache.jsonl"

# caching
# cache to jsonl
# load cache
class Embedding_Cache:
    def __init__(self):
        self.cache = self.load_embedding_cache()

    def load_embedding_cache(self):
        embedding_cache_path = Path(default_embedding_cache_path)
        if embedding_cache_path.exists():
            with open(embedding_cache_path, 'r') as json_file:
                json_list = list(json_file)
                
                json_list = [json.loads(json_str) for json_str in json_list]
        else:
            json_list = []
            

        return(json_list)
    
    def get_cache_as_df(self):
        cache_df = pd.DataFrame(self.cache)
        return(cache_df)
    
    def add_to_cache(self,embedding_dict):
        # writes to file
        with open(default_embedding_cache_path,'a') as f:
            f.write(json.dumps(embedding_dict) + "\n")
            
        # adds to cache in memory
        self.cache.append(embedding_dict)
        
# Functions to run on script load
cache = Embedding_Cache()

# embed function
# lazy built in rate limit handling
def __get_embedding(text: str, engine="text-embedding-ada-002", quiet = True) -> [float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    # try api, get response, if rate limit wait, else retry after delay,
    multiplier = 2
    sleepy_time = 1
    success_bool = False
    
    while not success_bool:
        try:
            embedding = openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]
            # reset sleepy time
            sleepy_time = 1
            return(embedding)
        except Exception as e:
            error_message = e.error['message'].lower()
            if not quiet:
                print(f"Error Message: {error_message}\nSleeping for {sleepy_time} seconds")
            if 'incorrect api key' in error_message:
                raise RuntimeError("incorrect api key")
            elif 'limit' in error_message:
                sleep(sleepy_time)
                sleepy_time = sleepy_time*multiplier
                
# make loading embeddings not dumb
def embed_string_list(text_list : [str], engine="text-embedding-ada-002") -> []:
    # check if text is already in cache
    # can optimize caching later
    if type(text_list) != list:
        raise TypeError("text list must be a list")
    text_df = pd.DataFrame({'text':text_list})
    
    cache_df = cache.get_cache_as_df()
    if cache_df.shape[0] > 0:
        text_df = pd.merge(text_df,cache_df, how = 'left', on = 'text')
    else:
        text_df['engine'] = engine
        text_df['vector'] = np.nan
    
    mask = text_df.vector.isna()


    # embed uncached items
    # note optimization here was done iteratively with openai's ratelimits in mind
    # I would do this differently if there was not the need to save information as it comes in
    for idx, row in text_df[mask].iterrows():
        text = row.text
        vector = __get_embedding(text,engine)
        embedding_dict = {'text':text,'engine': engine, 'vector': vector}
        embedded_df = pd.DataFrame([embedding_dict],index=[idx])
        text_df.update(embedded_df)
        # save cache on disk
        cache.add_to_cache(embedding_dict)

    return(text_df)
    
        
# cosine similarity
def cosine_similarity(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return(cos_sim)

def __label_score(embeddings, label_embeddings):
    return cosine_similarity(embeddings, label_embeddings[1]) - cosine_similarity(embeddings, label_embeddings[0])

# zero shot classification
def zero_shot_classification(text_list : [str],labels : [str, str], engine="text-embedding-ada-002"):
    # check for correct types
    if type(text_list) != list:
        raise TypeError("text_list must be a list")
    
    if type(labels) != list:
        raise TypeError("labels must be a list")

    if len(labels) != 2:
        raise ValueError("labels must be of length 2")

    # combine items in list to avoid caching twice
    text_list.extend(labels)
    text_df = embed_string_list(text_list)
    
    labels_df = text_df.iloc[-2:]
    to_label_df = text_df.iloc[0:-2]
    
    string_embeddings = to_label_df['vector']
    label_embeddings = to_label_df['vector']
    
    probas = [__label_score(item,label_embeddings) for item in string_embeddings]
    dict_list = []
    for idx, prob in enumerate(probas):
        if prob > 0:
            d = {'text':text_list[idx],'label':labels[1],'score':prob}
        else:
            d = {'text':text_list[idx],'label':labels[0],'score':prob}
        dict_list.append(d)
    return(dict_list)
        
        






