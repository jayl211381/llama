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
    file = "c4-train.00000-of-01024"
    STRUCT_PATH = f"{FOLDER_PATH}structure_count_{file}"
            
    # Load the text list
    with open(f"{FOLDER_PATH}{file}", "rb") as f:
        text_list = pickle.load(f) 
        
    # Load the exsisting structure count 
    if os.path.exists(STRUCT_PATH):
        # with open(STRUCT_PATH, "rb") as f:
        #     struct_dict = pickle.load(f) 
        raise Exception("File already exists, rerun to prevent dups")
    else:
        struct_dict = {"structure": [], "length": [], "count": [], "example": []}
        
    # Use punkt to detect sentences in a paragraph
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for texts in tqdm(text_list):
        text = texts['text']
        for sentence in sent_detector.tokenize(text.strip()):
            # https://www.nltk.org/api/nltk.tag.html
            # Penn Treebank tagset by defualt
            tokened = pos_tag(word_tokenize(sentence))
            struct_list = []
            for element in tokened:
                struct_list.append(element[1])
            # Append to the dictionary
            if struct_list not in struct_dict["structure"]:
                struct_dict["structure"].append(struct_list)
                struct_dict["example"].append(sentence)   
                struct_dict["count"].append(0) 
                struct_dict["length"].append(len(struct_list))
            else:
                index = struct_dict["structure"].index(struct_list)
                struct_dict["count"][index] += 1
        
    # Store the result
    with open(STRUCT_PATH, 'wb') as f:
        # stack, first in first out (FIFO)
        pickle.dump(struct_dict, f)      
          
    print("done")