import requests
import os
import zipfile

# GitHub API URL for the contents of the HGV23 folder
api_url = "https://api.github.com/repos/papyri/idp.data/zipball/main/HGV_meta_EpiDoc/HGV220"

# Directory to save the downloaded zip file
zip_file_path = "xml/HGV220.zip"
download_dir = "xml/HGV220"
os.makedirs(os.path.dirname(zip_file_path), exist_ok=True)

def download_file(url, path):
    response = requests.get(url)
    with open(path, 'wb') as file:
        file.write(response.content)

def unzip_file(zip_file, target_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def main():
    # Download the zip file
    print("Downloading zip file...")
    download_file(api_url, zip_file_path)
    print("Zip file downloaded.")

    # Unzip the file
    print("Extracting zip file...")
    unzip_file(zip_file_path, download_dir)
    print("Zip file extracted.")

    print("Download completed.")

if __name__ == "__main__":
    main()
