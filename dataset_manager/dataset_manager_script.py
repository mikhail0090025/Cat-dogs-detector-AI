import os
import io
import asyncio
import aiohttp
import aiofiles
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

async def download_dataset(session, url, filepath):
    async with session.get(url) as response:
        if response.status == 200:
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(await response.read())
            print(f"Downloaded {filepath}")
        else:
            raise Exception(f"Download failed with status {response.status}")

async def extract_zip(zip_path, extract_to):
    async with aiofiles.open(zip_path, 'rb') as f:
        content = await f.read()
        with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
            zip_ref.extractall(extract_to)
    print("Dataset extracted to:", extract_to)

async def process_images(folder, category):
    global images, outputs
    all_files = os.listdir(folder)
    max_files_count = 2000
    for i, filename in enumerate(all_files[:max_files_count]):
        try:
            path = os.path.join(folder, filename)
            print(f"Path: {path}\n{i+1}/{max_files_count}")
            img = Image.open(path)
            img = img.convert("RGB")
            img_resized = img.resize((30, 30), Image.Resampling.LANCZOS)
            img_array = ((np.array(img_resized) / 127.5) - 1).astype(np.float32)
            images.append(img_array)
            outputs.append([0] * 2)
            outputs[-1][category] = 1
            print("Dog image" if category == 0 else "Cat image")
        except UnidentifiedImageError:
            print(f"Error loading {path}: not an image, skipping")
            continue

async def get_images():
    global images, outputs, images_are_loaded

    try:
        if os.path.exists(os.path.join(current_dir, 'numpy_dataset.npz')):
            print("Dataset was found. Loading...")
            data = np.load(os.path.join(current_dir, 'numpy_dataset.npz'))
            images = data['images']
            outputs = data['outputs']
            images_are_loaded = True
            return

        print("Dataset was not found. Creating...")
        url = "https://www.kaggle.com/api/v1/datasets/download/bhavikjikadara/dog-and-cat-classification-dataset"
        zip_path = os.path.join(current_dir, file_name)

        if not os.path.exists(dataset_directory):
            if not os.path.exists(os.path.join(current_dir, file_name)):
                async with aiohttp.ClientSession() as session:
                    await download_dataset(session, url, zip_path)
            await extract_zip(zip_path, current_dir)

        all_folders = [
            os.path.join(dataset_directory, "Dog"),
            os.path.join(dataset_directory, "Cat"),
        ]

        tasks = [process_images(folder, all_folders.index(folder)) for folder in all_folders]
        await asyncio.gather(*tasks)

        images = np.array(images, dtype=np.float32)
        outputs = np.array(outputs, dtype=np.float32)
        indexes = np.random.permutation(len(images))
        images = images[indexes]
        outputs = outputs[indexes]
        images = np.array(images, dtype=np.float32)
        outputs = np.array(outputs, dtype=np.float32)
        np.savez_compressed('numpy_dataset.npz', images=images, outputs=outputs)
        images_are_loaded = True
    except Exception as e:
        print(f'Error while getting dataset has occurred: {e}')
        raise

def bytesToText(bytes:int):
    if bytes < 1024:
        return f'{bytes} Bytes.'
    elif bytes < 1024 * 1024:
        return f'{bytes / 1024:.2f} KB.'
    elif bytes < 1024 * 1024 * 1024:
        return f'{bytes / (1024 * 1024):.2f} MB.'
    elif bytes < 1024 * 1024 * 1024 * 1024:
        return f'{bytes / (1024 * 1024 * 1024):.2f} GB.'
    else:
        return f'{bytes / (1024 * 1024 * 1024 * 1024):.2f} TB.'

if __name__ == '__main__':
    asyncio.run(get_images())
    print(f'Images count: {len(images)}')
    print(f'Dataset size: {bytesToText((images.size * 4) + (outputs.size * 4))}.')
    print(f'Inputs shape: {images.shape}')
    print(f'Outputs shape: {outputs.shape}')
