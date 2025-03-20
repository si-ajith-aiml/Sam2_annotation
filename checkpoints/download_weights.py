import os
import requests

# URLs of the checkpoint files
urls = [
    # "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    # "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    # "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
]

# Define the save directory
save_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(save_dir, exist_ok=True)

# Function to download files
def download_file(url):
    filename = os.path.join(save_dir, os.path.basename(url))
    print(f"Downloading {filename} ...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    
    print(f"✅ Downloaded: {filename}")

# Download all files
for url in urls:
    download_file(url)

print("\n🎉 All checkpoints downloaded successfully!")
