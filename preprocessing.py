import os
import urllib
import urllib.request
import pathlib
from zipfile import ZipFile

import fashion_mnist_image_class

NAME = "fashion_mnist_images"
URL = f"https://nnfs.io/datasets/{NAME}.zip"
LOCATION = f"data/{NAME}"
FILE_PATH = f"{LOCATION}/{NAME}.zip"

def download():
  if not os.path.isfile(FILE_PATH):
    print(f'Downloading {URL} and saving as {FILE_PATH}...')
    pathlib.Path(LOCATION).mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, FILE_PATH)
  else:
    print(f"Skipping {URL} as {FILE_PATH} already exists!")

  print("Extracting...")
  with ZipFile(FILE_PATH) as zip_images:
    zip_images.extractall(f"{LOCATION}/extracted")
  print("Complete.")

def download_img(url_base, name):
  if not os.path.isfile(f"{LOCATION}/{name}"):
    print(f'Downloading {url_base}/{name} and saving as {LOCATION}/{name}...')
    pathlib.Path(LOCATION).mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{url_base}/{name}", f"{LOCATION}/{name}")

def load_dataset(root_path):
    dataset = []
    paths = [v for v in os.walk(root_path)]
    for path, _, names in paths[1:]:
        spath = path.split("/")
        classification = int(spath[-1])
        for name in names:
            # print(f"Adding (name: {name}, path: ./{path}, classification: {classification}, full_path: ./{path}/{name})")
            dataset.append(fashion_mnist_image_class.FashionMnistImageTensor(name, path, classification))

    return dataset

