import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random
import pickle

def copy_file(src, dst):
    shutil.copy2(src, dst)

def copy_files_to_destination(src_files, dst_folder, max_workers=4):
    os.makedirs(dst_folder, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for src_file in src_files:
            dst_file = os.path.join(dst_folder, os.path.basename(src_file))
            futures.append(executor.submit(copy_file, src_file, dst_file))

        for future in tqdm(futures, total=len(src_files), desc="Copying files", unit="file"):
            future.result()

if __name__ == "__main__":
    with open('tuh_files.pkl', 'rb') as f:
        files = pickle.load(f)
    
    random.shuffle(files)
        
    source_files = files[:10000]  # Replace with your list of files
    destination_folder = "/scratch/s194260/tuh_eeg"

    copy_files_to_destination(source_files, destination_folder)