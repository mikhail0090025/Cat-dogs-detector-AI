import aiohttp
import aiofiles
import numpy as np
import asyncio
import tensorflow as tf
import keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.graph_objects as go
import json

path_to_model = os.path.join(os.path.dirname(__file__), 'model.keras')
print(f"Path to model: {path_to_model}")

'''
all_losses = []
all_val_losses = []
all_accuracies = []
all_val_accuracies = []
main_model = None
train_generator = None
val_generator = None
lr_scheduler = None
SaveCheckpoint = None
dataset_was_got = False'''

# Save statistic
def save_metrics():
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies
    metrics = {
        'all_losses': all_losses,
        'all_val_losses': all_val_losses,
        'all_accuracies': all_accuracies,
        'all_val_accuracies': all_val_accuracies
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Metrics saved to metrics.json")

# Load statistic
def load_metrics():
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
            all_losses = metrics.get('all_losses', [])
            all_val_losses = metrics.get('all_val_losses', [])
            all_accuracies = metrics.get('all_accuracies', [])
            all_val_accuracies = metrics.get('all_val_accuracies', [])
        print("Metrics loaded from metrics.json")

# Create or load model
def get_model():
    try:
        if tf.io.gfile.exists(path_to_model):
            model = keras.models.load_model(path_to_model)
            print("Model returned")
            return model
        else:
            print('Model not found, creating a new one')

        model = tf.keras.models.Sequential([
            # Первый сверточный блок
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(50, 50, 3), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            keras.layers.Dropout(0.3),

            # Второй сверточный блок
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            keras.layers.Dropout(0.3),

            # Третий сверточный блок
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            keras.layers.Dropout(0.3),

            # Переход к полносвязной части
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        tf.keras.models.save_model(model, path_to_model, overwrite=True, include_optimizer=True, save_format=None)
        print("Model returned")
        return model
    except Exception as e:
        print(f"Error while loading NN happened: {e}")
        raise

def prepare_dataset(images, outputs):
    global train_generator, val_generator, lr_scheduler, SaveCheckpoint
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        zoom_range=0.25,
        fill_mode='nearest',
        validation_split=0.15
    )
    val_datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow(images, outputs, batch_size=64, subset='training', shuffle=True)
    val_generator = val_datagen.flow(images, outputs, batch_size=64, subset='validation', shuffle=False)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3, min_lr=1e-6)
    SaveCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        path_to_model,
        monitor='accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )

def go_epochs(epochs_count):
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies, main_model, train_generator, val_generator
    if main_model is None or train_generator is None or val_generator is None:
        raise ValueError("Model or generators not initialized. Run main() first.")
    load_metrics()
    print("GO EPOCHS CALLED")
    print(f"main_model: {main_model}")
    history = main_model.fit(
        train_generator,
        epochs=epochs_count,
        validation_data=val_generator,
        callbacks=[SaveCheckpoint, lr_scheduler]
    )
    all_losses.extend(history.history['loss'])
    all_val_losses.extend(history.history['val_loss'])
    all_accuracies.extend(history.history['accuracy'])
    all_val_accuracies.extend(history.history['val_accuracy'])
    print("Updated metrics:", len(all_losses), len(all_val_losses), len(all_accuracies), len(all_val_accuracies))
    save_metrics()

def get_graphic(losses, val_losses, accuracies, val_accuracies):
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies
    load_metrics()
    if not all([losses, val_losses, accuracies, val_accuracies]):
        raise ValueError("Metrics are empty. Call 'go_epochs' at least once.")
    lengths = [len(losses), len(val_losses), len(accuracies), len(val_accuracies)]
    if len(set(lengths)) > 1:
        raise ValueError(f"Length of lists of metrics is different: {lengths}")
    fig = go.Figure(
        data=[
            # go.Scatter(x=list(range(len(losses))), y=losses, name='Train Loss', line=dict(color='blue')),
            # go.Scatter(x=list(range(len(val_losses))), y=val_losses, name='Validation Loss', line=dict(color='red')),
            go.Scatter(x=list(range(len(accuracies))), y=accuracies, name='Train Accuracy', line=dict(color='green')),
            go.Scatter(x=list(range(len(val_accuracies))), y=val_accuracies, name='Validation Accuracy', line=dict(color='orange')),
        ],
        layout_title_text="Training Statistics"
    )
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Value",
        yaxis_range=[min(min(accuracies), min(val_accuracies)), 1],
        template='plotly_dark'
    )
    return fig

async def fetch_data(session, url, filepath):
    async with session.get(url, timeout=10, ssl=False) as response:
        response.raise_for_status()
        data = await response.read()
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(data)
        return filepath

async def main():
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies
    global main_model, train_generator, val_generator, lr_scheduler, SaveCheckpoint, dataset_was_got

    print("main() called")
    # Сброс переменных
    all_losses = []
    all_val_losses = []
    all_accuracies = []
    all_val_accuracies = []
    main_model = None
    train_generator = None
    val_generator = None
    lr_scheduler = None
    SaveCheckpoint = None
    dataset_was_got = False

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
                prepare_dataset(images, outputs)
                main_model = get_model()
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

if __name__ == '__main__':
    asyncio.run(main())