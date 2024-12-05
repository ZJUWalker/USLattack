data_root = '/home/lzh/projects/SplitLLM/data'

dataset_cache_dir = f'{data_root}/datasets/'
model_download_dir = f'{data_root}/models/'
model_cache_dir = f'{data_root}/cache/'
attacker_path = f'{data_root}/models/attacker/'
mapper_path = f'{data_root}/models/mapper/'
reducer_path = f'{data_root}//models/reducer/'
lora_path = f'{data_root}/models/lora/'

dxp_moe_range = {0.08, 0.21, 0.38}
gaussian_moe_range = {3.0, 5.0, 8.0}
dc_moe_range = {8.0, 32.0, 64.0}