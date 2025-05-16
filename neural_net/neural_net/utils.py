import os
import subprocess
import zipfile
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests

images_response = requests.get('http://localhost:8000/get_images')
outputs_response = requests.get('http://localhost:8000/get_outputs')

images_response.raise_for_status()
outputs_response.raise_for_status()

images_json = images_response.json()
outputs_json = outputs_response.json()

print(outputs_json)
print(type(outputs_json))