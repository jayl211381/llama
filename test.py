import torch
import subprocess
import os

os.system("python --version")  
# os.system("cmd 1 | another_command > output_file")  
# subprocess.run(["ls", "-l"]) 
subprocess.run(["python", "--version"]) 

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.cuda.current_device())

print(torch.cuda.device(0))

print(torch.cuda.get_device_name(0))

print("done")