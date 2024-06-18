import os
import requests
import xml.etree.ElementTree as ET

def find_graphic_url_in_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    graphic_urls = []
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}  # Define the namespace mapping
    for figure in root.findall('.//tei:figure', ns):
        graphic = figure.find('./tei:graphic', ns)
        url = graphic.get('url')
        if url and url.startswith("https://digi.ub.uni-heidelberg.de/diglit"):
            graphic_urls.append(url)

    return graphic_urls

def download_image(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {save_path}")
    else:
        print(f"Failed to download {url}")

def process_xml_files(xml_folder, image_save_folder):
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)
    
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            
            xml_file_path = os.path.join(xml_folder, filename)
            graphic_urls = find_graphic_url_in_xml(xml_file_path)
            
            for url in graphic_urls:
                url_parts = url.split('/')
                image_name = url_parts[-1]
                image_url = f"{url}/0001/_image"
                filename = os.path.splitext(filename)[0]
                save_path = os.path.join(image_save_folder, f"{filename}_{image_name}.jpg")
                download_image(image_url, save_path)

if __name__ == "__main__":
    xml_folder = '/Users/cprao/Desktop/heidelberg_SEM3/practical/papyri-matching/xml/HGV23'  # Update with the path to your folder containing XML files
    image_save_folder = '/Users/cprao/Desktop/heidelberg_SEM3/practical/papyri-matching/images/HGV23_images'  # Update with the path to the folder where you want to save images
    process_xml_files(xml_folder, image_save_folder)
