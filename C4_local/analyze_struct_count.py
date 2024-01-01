import json
import pickle
import nltk 
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm

# nltk.download('averaged_perceptron_tagger') 
# nltk.download('punkt')
    
'''
Generate word sentence structure count from C4 dataset
Hypothesis: Most of conversations/descriptions use similar sentence structures
- Formulate a fixed embedding of structure relationships
- Replace the positional encoding layer with the structure embedding
'''

if __name__ == '__main__':

    FOLDER_PATH = "C:/Users/Jayl2/OneDrive/llama/C4_local/data_pickle/"
    file = "structure_count_c4-train.00000-of-01024_temp"
    
    # Load the text list
    with open(f"{FOLDER_PATH}{file}", "rb") as f:
        struct_dict1 = pickle.load(f) 
        
    # Check the top 10 most common structures
    temp_count = struct_dict1['count'].copy()
    temp_count.sort(reverse=True)
    
    # Get the top10 most common structure indexes
    top10_index = []
    for count in temp_count[0:10]:
        top10_index.append(struct_dict1['count'].index(count))
    
    # Get the top10 
    for index in top10_index:
        print(struct_dict1['structure'][index])
        print(struct_dict1['count'][index])
        print(struct_dict1['example'][index])
        print("------------------")
    print("done")