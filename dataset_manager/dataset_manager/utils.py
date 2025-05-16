import kagglehub
import os
import subprocess
import zipfile
import numpy as np
from PIL import Image, UnidentifiedImageError

current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'dog-and-cat-classification-dataset.zip'
dataset_directory = os.path.join(current_dir, 'PetImages')

np.random.seed(42)
images_are_loaded = False
images = []
outputs = []

# Download and extract learning dataset if its not
if not os.path.exists(dataset_directory):
    if not os.path.exists(os.path.join(current_dir, file_name)):
        command = f'''curl -L -o {current_dir}/{file_name}  https://www.kaggle.com/api/v1/datasets/download/bhavikjikadara/dog-and-cat-classification-dataset'''
        print(command)
        downloading_process = subprocess.run(command, shell=True)
        print(downloading_process)

    with zipfile.ZipFile(os.path.join(current_dir, file_name), 'r') as zip_ref:
        zip_ref.extractall(current_dir)
    print("Dataset extracted to:", current_dir)

all_folders = [
    os.path.join(dataset_directory, "Cat"),
    os.path.join(dataset_directory, "Dog"),
]

def get_images():
    global images, outputs, images_are_loaded

    if os.path.exists(os.path.join(current_dir, 'numpy_dataset.npz')):
        print("Dataset was found. Loading...")
        data = np.load(os.path.join(current_dir, 'numpy_dataset.npz'))
        images = data['images']
        outputs = data['outputs']
        images_are_loaded = True
        return
    
    print("Dataset was not found. Creating...")

    for folder in all_folders:
        all_files = os.listdir(folder)
        max_files_count = 2000
        for i, filename in enumerate(all_files[:max_files_count]):
            try:
                path = os.path.join(folder, filename)
                # print(f"Path: {path}\n{i+1}/{max_files_count}")
                img = Image.open(path)
                img = img.convert("RGB")
                img_resized = img.resize((40, 40), Image.Resampling.LANCZOS)
                img_array = (np.array(img_resized) / 127.5) - 1
                images.append(img_array)
                outputs.append([0] * 2)
                # Определяем класс по имени файла
                if "cat" in filename.lower():
                    outputs[-1][0] = 1  # Cat
                elif "dog" in filename.lower():
                    outputs[-1][1] = 1  # Dog
            except UnidentifiedImageError:
                print(f"Error loading {path}: not an image, skipping")
                continue

    images = np.array(images)
    outputs = np.array(outputs)
    indexes = np.random.permutation(len(images))
    images = images[indexes]
    outputs = outputs[indexes]
    np.savez_compressed('numpy_dataset.npz', images=images, outputs=outputs)
    images_are_loaded = True

get_images()
print(f'Images count: {len(images)}')
print(f'Inputs size: {images.shape[0]*images.shape[1]*images.shape[2]*images.shape[3]*4} bytes.')
print(f'Inputs shape: {images.shape}')
print(f'Outputs shape: {outputs.shape}')