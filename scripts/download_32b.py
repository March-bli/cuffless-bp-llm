import sys
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-32B', local_dir='/mnt/parscratch/users/acp25bl/models/Qwen3-32B')
print('MODEL_32B_DOWNLOAD_DONE')
