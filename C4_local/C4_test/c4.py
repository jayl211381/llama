import requests
import json
import pickle
import os

if __name__ == "__main__":
    
    FOLDER_PATH = "C:\Users\Jayl2\OneDrive\llama\C4_local\C4_test\db"
    
    # No more than 100 at a time for api
    START = 1
    FINISH = 100
    
    db_path = f"{FOLDER_PATH}/c4_{START}_{FINISH}"
    
    # Check for duplicates
    if os.path.exists(db_path):
        raise ValueError("File already exists")
    
    # MAX training - 2.21M
    # https://huggingface.co/datasets/c4
    response = requests.get(f"https://datasets-server.huggingface.co/rows?dataset=c4&config=en&split=train&offset={START}&length={FINISH}")

    content = response.content
    decoded_string = content.decode("utf-8")

    res_dict = json.loads(decoded_string)

    # Store the result
    with open(db_path, 'wb') as f:
        # stack, first in first out (FIFO)
        pickle.dump(res_dict, f)      

    print("done")