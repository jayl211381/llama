# python -m torch.distributed.run --nproc_per_node 1 example_chat_completion.py \--ckpt_dir llama-2-7b-chat/ \--tokenizer_path tokenizer.model \--max_seq_len 512 --max_batch_size 6

# torchrun --nproc_per_node 1 example_chat_completion.py \--ckpt_dir llama-2-7b-chat/ \--tokenizer_path tokenizer.model \--max_seq_len 512 --max_batch_size 6

# torchrun --nproc_per_node 1 example_chat_completion.py \
#     --ckpt_dir llama-2-7b-chat/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 512 --max_batch_size 6

torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir llama-2-7b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 --max_batch_size 4

# For git bash
# conda activate c:/Users/Jayl2/OneDrive/llama/.conda

# For path of torchrun 
# https://stackoverflow.com/questions/77425569/llama2-running-pytorch-produces-a-failed-to-create-process

# In generation.py, set nccl to gloo
# "env": {"PL_TORCH_DISTRIBUTED_BACKEND":"gloo"},