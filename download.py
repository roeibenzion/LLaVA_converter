import os
import requests
from tqdm import tqdm
# Function to download the zip file (without extraction)
def download(url, download_folder):
    local_zip_path = os.path.join(download_folder, os.path.basename(url))

    if not os.path.exists(local_zip_path):  # Check if the zip file already exists
        # Download the zip file with a progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB block size for faster downloads

        with open(local_zip_path, 'wb') as zip_file:
            for data in tqdm(response.iter_content(block_size), total=total_size // block_size, unit='KB', desc=f"Downloading {os.path.basename(url)}"):
                zip_file.write(data)

        print(f"{os.path.basename(url)} downloaded to {download_folder}.")
    else:
        print(f"{os.path.basename(url)} already exists, skipping download.")


from huggingface_hub import hf_hub_download, list_repo_files
import os

# Define repo_id and the download folder
repo_id = "liuhaotian/LLaVA-Pretrain"
download_folder = "./playground/data/LLaVA-Pretrain"
os.makedirs(download_folder, exist_ok=True)

# List all files in the dataset repository
file_list = list_repo_files(repo_id, repo_type="dataset")
print(f"Found {len(file_list)} files in the dataset.")

# Download each file using hf_hub_download
for filename in file_list:
    print(f"Downloading {filename}...")
    file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=download_folder, force_download=True)
    print(f"Downloaded {filename} to {file_path}")

print("All files downloaded successfully.")

