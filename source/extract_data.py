import tarfile
import os
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def extract_tar_file(tar_path, extract_to):
    """Extracts a .tar file into the specified directory."""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_to)
        print(f"Extracted {tar_path} to {extract_to}")

if __name__ == "__main__":
    tar_file_path = "data/data.tar"  # Update this path to your .tar file
    extracted_data_dir = "data/raw"
    extract_tar_file(tar_file_path, extracted_data_dir)
