import requests
import os

# GitHub API URL for the contents of the HGV23 folder
api_url = "https://api.github.com/repos/papyri/idp.data/contents/HGV_meta_EpiDoc/HGV875"

# Directory to save the downloaded files
download_dir = "xml/HGV875"
os.makedirs(download_dir, exist_ok=True)

def download_file(url, path):
    response = requests.get(url)
    with open(path, 'wb') as file:
        file.write(response.content)

def main():
    # Fetch the list of files
    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file['type'] == 'file':
                file_url = file['download_url']
                file_name = file['name']
                file_path = os.path.join(download_dir, file_name)
                
                print(f"Downloading {file_name}...")
                download_file(file_url, file_path)
        print("Download completed.")
    else:
        print(f"Failed to fetch file list: {response.status_code}")

if __name__ == "__main__":
    main()
