import json
import pickle

if __name__ == '__main__':
    
    FOLDER_PATH = "C:/Users/Jayl2/OneDrive/llama/C4_local/"
    db_path = f"{FOLDER_PATH}data_pickle/c4-train.00000-of-01024"
    
    json_file = "c4-train.00000-of-01024.json"

    # Iterate over each json file
    text_list = []
    with open(f"{FOLDER_PATH}data_json/{json_file}",encoding="utf8") as file:
        for line in file:
            text_list.append(json.loads(line))
            
    # Store the result
    with open(db_path, 'wb') as f:
        # stack, first in first out (FIFO)
        pickle.dump(text_list, f)    
                  
    # Closing file
    f.close()