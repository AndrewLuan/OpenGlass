import os
import time

import requests
from bs4 import BeautifulSoup


def download_image(url, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    response = requests.get(url)
    file_name = os.path.join(folder_path, url.split("/")[-1])
    with open(file_name, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded {file_name}")


def download_images_from_webpage(webpage_url, folder_path, known_images):
    response = requests.get(webpage_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')
    img_urls = [img['src'] for img in img_tags if 'src' in img.attrs]

    new_images = []
    for img_url in img_urls:
        if not img_url.startswith('http'):
            img_url = webpage_url + img_url
        if img_url not in known_images:
            download_image(img_url, folder_path)
            new_images.append(img_url)

    return new_images


def monitor_images(webpage_url, folder_path, interval=60):
    known_images = set()

    while True:
        new_images = download_images_from_webpage(
            webpage_url, folder_path, known_images)
        known_images.update(new_images)
        time.sleep(interval)


webpage_url = 'http://localhost:8081'  # 替换为你想要下载图片的网页 URL
folder_path = './downloaded_images'      # 图片保存的文件夹
interval = 1  # 爬取间隔时间（秒）

monitor_images(webpage_url, folder_path, interval)
