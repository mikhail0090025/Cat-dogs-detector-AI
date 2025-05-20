import aiohttp
import aiofiles
import numpy as np
import asyncio
import tensorflow as tf
import keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.graph_objects as go

# Learning data
train_generator = None
val_generator = None
lr_scheduler = None
SaveCheckpoint = None
main_model = None

# Statistics
all_losses = []
all_val_losses = []
all_accuracies = []
all_val_accuracies = []

path_to_model = os.path.join(os.path.dirname(__file__), 'model.keras')
print(f"Path to model: {path_to_model}")

# Create or load model
def get_model():
    try:
        if tf.io.gfile.exists(path_to_model):
            model = keras.models.load_model(path_to_model)
            return model
        else:
            print('Model not found, creating a new one')

        model = tf.keras.models.Sequential([
            keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(40, 40, 3)),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

            keras.layers.Conv2D(64, (3,3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

            keras.layers.Conv2D(128, (3,3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

            keras.layers.Flatten(),

            keras.layers.Dense(512, activation="relu"),

            keras.layers.Dense(256, activation="relu"),

            keras.layers.Dense(64, activation="relu"),

            keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        model.summary()
        tf.keras.models.save_model(
            model, path_to_model, overwrite=True,
            include_optimizer=True, save_format=None)
        return model
    except Exception as e:
        print(f"Error while loading NN happened: {e}")

def prepare_dataset(images, outputs):
    global train_generator, val_generator, lr_scheduler, SaveCheckpoint
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.9, 1.1],
        horizontal_flip=False,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow(
        images,
        outputs,
        batch_size=512,
        subset='training',
        shuffle=True
    )
    val_generator = val_datagen.flow(
        images,
        outputs,
        batch_size=512,
        subset='validation',
        shuffle=False
    )
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy', factor=0.5, patience=3, min_lr=1e-6
    )
    SaveCheckpoint = tf.keras.callbacks.ModelCheckpoint(path_to_model,
        monitor='accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch')

def go_epochs(model, epochs_count):
    history = model.fit(
        train_generator,
        epochs=epochs_count,
        validation_data=val_generator,
        callbacks=[SaveCheckpoint, lr_scheduler]
    )
    all_losses.extend(history.history['loss'])
    all_val_losses.extend(history.history['val_loss'])
    all_accuracies.extend(history.history['accuracy'])
    all_val_accuracies.extend(history.history['val_accuracy'])

def get_graphic():
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies
    fig = go.Figure(
        data=[
            go.Bar(y=all_losses, name='Train Loss'),
            go.Bar(y=all_val_losses, name='Validation Loss'),
            go.Bar(y=all_accuracies, name='Train Accuracy'),
            go.Bar(y=all_val_accuracies, name='Validation Accuracy'),
        ],

        layout_title_text="Statistics"
    )

    return fig

async def fetch_data(session, url, filepath):
    async with session.get(url, timeout=10, ssl=False) as response:
        response.raise_for_status()
        data = await response.read()
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(data)
        return filepath

dataset_was_got = False
async def main():
    global main_model
    main_model = get_model()
    # Getting dataset from 'dataset_manager' microservice
    async with aiohttp.ClientSession() as session:
        global dataset_was_got
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
                prepare_dataset(images, outputs)
                dataset_was_got = True
                break
            except aiohttp.client_exceptions.ClientConnectorError as e:
                print('Couldnt connect. Retry in 10 seconds...')
                await asyncio.sleep(10)
            except Exception as e:
                print(f'Unexpected error has occurred: {e}')
                raise
    if not dataset_was_got:
        print("Dataset wasnt got.")
        exit(1)
    go_epochs(main_model, 20)

if __name__ == '__main__':
    asyncio.run(main())