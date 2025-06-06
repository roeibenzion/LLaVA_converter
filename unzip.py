import zipfile
import os
from tqdm import tqdm

zip_path = './playground/data/LLaVA-Pretrain/images.zip'
extract_path = './playground/data/LLaVA-Pretrain/images'

# Create the directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Open the zip file and get the list of files for tqdm
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    for file in tqdm(file_list, desc="Extracting files", unit="file"):
        zip_ref.extract(file, extract_path)

print(f"Extracted to {extract_path}")