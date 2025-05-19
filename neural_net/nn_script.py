import aiohttp
import aiofiles
import numpy as np
import asyncio
import tensorflow as tf
import keras

path_to_model = 'model'

def get_model():
    if tf.io.gfile.exists(path_to_model):
        model = tf.saved_model.load(path_to_model)
        return model
    else:
        print('Model not found, creating a new one')

    model = tf.keras.models.Sequential([
        keras.layers.Conv2D(32, (3,3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        keras.layers.Conv2D(64, (3,3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        keras.layers.Conv2D(128, (3,3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

        keras.layers.Flatten(),

        keras.layers.Dense(2048, activation="relu"),

        keras.layers.Dense(256, activation="relu"),

        keras.layers.Dense(64, activation="relu"),

        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

async def fetch_data(session, url, filepath):
    async with session.get(url, timeout=10, ssl=False) as response:
        response.raise_for_status()
        data = await response.read()
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(data)
        return filepath

async def main():
    async with aiohttp.ClientSession() as session:
        for i in range(60):
            print(f'Attempt {i}')
            try:
                images_file = await fetch_data(session, 'http://dataset_manager:5000/get_images', 'images.npz')
                outputs_file = await fetch_data(session, 'http://dataset_manager:5000/get_outputs', 'outputs.npz')

                data_images = np.load(images_file)
                data_outputs = np.load(outputs_file)
                images = data_images['images']
                outputs = data_outputs['outputs']

                print('Images shape:', images.shape)
                print('Outputs shape:', outputs.shape)
                print('Dataset was got')
                break
            except aiohttp.client_exceptions.ClientConnectorError as e:
                print('Couldnt connect. Retry in 10 seconds...')
                await asyncio.sleep(10)
            except Exception as e:
                print(f'Unexpected error has occurred: {e}')
                raise

if __name__ == '__main__':
    asyncio.run(main())