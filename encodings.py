import numpy as np
from utils import convert_to_str
from sentence_transformers import SentenceTransformer
import pandas as pd

# Convert a DataFrame to a sb representation.
def sb_encoding(df):
    
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    x_str = np.zeros((len(df), 384))
    
    encoded_str = encoder.encode(df.apply(convert_to_str, axis = 1).values)
    
    for i in range(len(df)):
        x_str[i] = encoded_str[i]
    
    return x_str

# Converts categorical features to integer codes.
def int_encoding(df):
        
    for feat in df.columns:
        df.loc[:, feat] = pd.Categorical(df.loc[:, feat]).codes
    
    return df.values