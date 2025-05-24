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
from tensorflow.keras.optimizers import Adam

path_to_model = os.path.join(os.path.dirname(__file__), 'model.keras')
print(f"Path to model: {path_to_model}")

# Глобальные переменные
all_losses = []
all_val_losses = []
all_accuracies = []
all_val_accuracies = []
all_learning_rates = []
main_model = None
train_generator = None
val_generator = None
SaveCheckpoint = None
lr_scheduler = None
dataset_was_got = False

# Сохранение метрик
def save_metrics():
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies, all_learning_rates
    metrics = {
        'all_losses': all_losses,
        'all_val_losses': all_val_losses,
        'all_accuracies': all_accuracies,
        'all_val_accuracies': all_val_accuracies,
        'all_learning_rates': all_learning_rates,
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Metrics saved to metrics.json")

# Загрузка метрик
def load_metrics():
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies, all_learning_rates
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
            all_losses = metrics.get('all_losses', [])
            all_val_losses = metrics.get('all_val_losses', [])
            all_accuracies = metrics.get('all_accuracies', [])
            all_val_accuracies = metrics.get('all_val_accuracies', [])
            all_learning_rates = metrics.get('all_learning_rates', [])
        print("Metrics loaded from metrics.json")

# Создание или загрузка модели
def get_model():
    try:
        if tf.io.gfile.exists(path_to_model):
            model = keras.models.load_model(path_to_model)
            print("Model returned")
            return model
        else:
            print('Model not found, creating a new one')

        model = tf.keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(70, 70, 3)),
            # keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            # keras.layers.Dropout(0.3),

            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            # keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            # keras.layers.Dropout(0.3),

            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            # keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            # keras.layers.GlobalAveragePooling2D(),
            keras.layers.Flatten(),
            # keras.layers.Dropout(0.3),

            keras.layers.Dense(2048, activation="relu"),
            # keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.4),

            keras.layers.Dense(256, activation="relu"),
            # keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.4),

            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            # keras.layers.BatchNormalization(),
            keras.layers.Dense(2, activation='softmax')
        ])
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])
        model.summary()
        tf.keras.models.save_model(model, path_to_model, overwrite=True, include_optimizer=True)
        print("Model returned")
        return model
    except Exception as e:
        print(f"Error while loading NN happened: {e}")
        raise

# Подготовка датасета
def prepare_dataset(images, outputs):
    global train_generator, val_generator, SaveCheckpoint, lr_scheduler

    # Strong augmentation
    # datagen = ImageDataGenerator(
    #     rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     brightness_range=[0.85, 1.15],
    #     horizontal_flip=True,
    #     zoom_range=0.25,
    #     shear_range=0.2,
    #     fill_mode='nearest',
    #     validation_split=0.2
    # )

    # Weak augmentation
    # datagen = ImageDataGenerator(
    #     rotation_range=10,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     brightness_range=[0.95, 1.05],
    #     horizontal_flip=True,
    #     zoom_range=0.1,
    #     shear_range=0.1,
    #     fill_mode='nearest',
    #     validation_split=0.2
    # )

    # Very weak augmentation
    # datagen = ImageDataGenerator(
    #     rotation_range=1,
    #     width_shift_range=0.01,
    #     height_shift_range=0.01,
    #     brightness_range=[0.99, 1.01],
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     zoom_range=0.01,
    #     shear_range=0.01,
    #     fill_mode='nearest',
    #     validation_split=0.2
    # )

    # Only flip augmentation
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )

    # No augmentation
    # datagen = ImageDataGenerator(validation_split=0.2)
    val_datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow(images, outputs, batch_size=32, subset='training', shuffle=True)
    val_generator = val_datagen.flow(images, outputs, batch_size=32, subset='validation', shuffle=False)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.9,
        patience=10,
        min_lr=1e-8
    )

    SaveCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        path_to_model,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )

# Обучение модели
def go_epochs(epochs_count):
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies, all_learning_rates, main_model, train_generator, val_generator, lr_scheduler
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
    print(history.history.keys())
    all_losses.extend([float(loss) for loss in history.history['loss']])
    all_val_losses.extend([float(val_loss) for val_loss in history.history['val_loss']])
    all_accuracies.extend([float(acc) for acc in history.history['accuracy']])
    all_val_accuracies.extend([float(val_acc) for val_acc in history.history['val_accuracy']])
    all_precisions.extend([float(acc) for acc in history.history['precision']])
    all_val_precisions.extend([float(val_acc) for val_acc in history.history['val_precision']])
    all_recalls.extend([float(acc) for acc in history.history['recall']])
    all_val_recalls.extend([float(val_acc) for val_acc in history.history['val_recall']])
    
    all_learning_rates.extend([float(main_model.optimizer.learning_rate.numpy()) for _ in range(epochs_count)])
    save_metrics()

# Построение графиков
def get_graphics(losses, val_losses, accuracies, val_accuracies, learning_rates, precisions, val_precisions, recalls, val_recalls):
    load_metrics()
    if not all([losses, val_losses, accuracies, val_accuracies, learning_rates]):
        raise ValueError("Metrics are empty. Call 'go_epochs' at least once.")
    lengths = [len(losses), len(val_losses), len(accuracies), len(val_accuracies)]
    if len(set(lengths)) > 1:
        raise ValueError(f"Length of lists of metrics is different: {lengths}")
    fig_accuracy = go.Figure(
        data=[
            go.Scatter(x=list(range(len(accuracies))), y=accuracies, name='Train Accuracy', line=dict(color='green')),
            go.Scatter(x=list(range(len(val_accuracies))), y=val_accuracies, name='Validation Accuracy', line=dict(color='orange')),
            # go.Scatter(x=list(range(len(precisions))), y=precisions, name='Train Precision', line=dict(color='red')),
            # go.Scatter(x=list(range(len(val_precisions))), y=val_precisions, name='Validation Precision', line=dict(color='blue')),
            # go.Scatter(x=list(range(len(recalls))), y=recalls, name='Train Recalls', line=dict(color='yellow')),
            # go.Scatter(x=list(range(len(val_recalls))), y=val_recalls, name='Validation Recalls', line=dict(color='black')),
        ],
        # layout_title_text="Accuracy / Precision / Recall"
        layout_title_text="Accuracy"
    )
    fig_loss = go.Figure(
        data=[
            go.Scatter(x=list(range(len(losses))), y=losses, name='Train Loss', line=dict(color='blue')),
            go.Scatter(x=list(range(len(val_losses))), y=val_losses, name='Validation Loss', line=dict(color='red')),
        ],
        layout_title_text="Loss"
    )
    fig_lr = go.Figure(
        data=[
            go.Scatter(x=list(range(len(learning_rates))), y=learning_rates, name='Learning Rate', line=dict(color='green')),
        ],
        layout_title_text="Learning Rate"
    )
    fig_accuracy.update_layout(xaxis_title="Epoch", yaxis_title="Value", yaxis_range=[min(min(accuracies), min(val_accuracies)), 1], template='plotly_dark')
    fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Value", yaxis_range=[0, max(max(losses), max(val_losses))], template='plotly_dark')
    fig_lr.update_layout(xaxis_title="Epoch", yaxis_title="Value", yaxis_range=[0, max(learning_rates)], template='plotly_dark')
    return fig_accuracy, fig_loss, fig_lr

# Асинхронная загрузка данных
async def fetch_data(session, url, filepath):
    async with session.get(url, timeout=10, ssl=False) as response:
        response.raise_for_status()
        data = await response.read()
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(data)
        return filepath

# Основная функция
async def main():
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies, all_learning_rates, all_precisions, all_val_precisions, all_recalls, all_val_recalls
    global main_model, train_generator, val_generator, SaveCheckpoint, dataset_was_got, lr_scheduler

    print("main() called")
    all_losses = []
    all_val_losses = []
    all_accuracies = []
    all_val_accuracies = []
    all_precisions = []
    all_val_precisions = []
    all_recalls = []
    all_val_recalls = []
    all_learning_rates = []
    main_model = None
    train_generator = None
    val_generator = None
    SaveCheckpoint = None
    dataset_was_got = False
    lr_scheduler = None

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