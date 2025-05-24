from flask import Flask, Response, request, jsonify
import numpy as np
import nn_script
from nn_script import go_epochs, main
from flask_cors import CORS
import asyncio
import requests
import json
import aiohttp
from urllib.parse import quote
import time

app = Flask(__name__)

CORS(app)

@app.route('/', methods=['GET'])
def root():
    return {'Response': 'This is a root of neural net'}, 200

@app.route('/pass_epochs', methods=['POST'])
def pass_epochs():
    try:
        data = request.get_json()
        if not data:
            return {'Response': 'No JSON data provided'}, 400
        epochs_count = data.get('epochs_count')
        print(f"epochs_count: {epochs_count}")

        if epochs_count is None:
            return {'Response': f'Epochs count was none'}, 400
        if type(epochs_count) is not int:
            return {'Response': f'Epochs count was not an integer ({type(epochs_count).__name__}, {epochs_count})'}, 400
        if epochs_count <= 0:
            return {'Response': f'Epochs count cant be zero or less, it should be at least 1 ({epochs_count})'}, 400
        go_epochs(epochs_count)
        return {'Response': f'{epochs_count} epochs have successfully run!'}, 200
    except Exception as e:
        return {'Response': f'Unexpected error has occured: {e}, ({type(e)})'}, 500

@app.route('/time_for_epoch', methods=['GET'])
def time_for_epoch():
    try:
        time1 = time.time_ns()
        go_epochs(1)
        time2 = time.time_ns()
        diff = time2 - time1
        diff_miliseconds = diff / 1000000.0
        if diff_miliseconds > 1000:
            return {'Response': f'One epoch has taken {round(diff_miliseconds / 1000.0, 2)} seconds'}, 200
        return {'Response': f'One epoch has taken {diff_miliseconds} milliseconds'}, 200
    except Exception as e:
        return {'Response': f'Unexpected error has occured: {e}, ({type(e)})'}, 500

@app.route('/inputs', methods=['GET'])
def inputs():
    return {'Response': nn_script.images}, 200

@app.route('/outputs', methods=['GET'])
def outputs():
    return {'Response': nn_script.outputs}, 200

@app.route('/graphic', methods=['GET'])
def graphic():
    try:
        print(nn_script.all_losses)
        fig_accuracy, fig_loss, fig_lr = nn_script.get_graphics(
            nn_script.all_losses, nn_script.all_val_losses, nn_script.all_accuracies, nn_script.all_val_accuracies, nn_script.all_learning_rates,
            nn_script.all_precisions, nn_script.all_val_precisions, nn_script.all_recalls, nn_script.all_val_recalls)
        accuracy_json = fig_accuracy.to_json()
        loss_json = fig_loss.to_json()
        lr_json = fig_lr.to_json()
        return {
            'accuracy_json': accuracy_json,
            'loss_json': loss_json,
            'lr_json': lr_json,
        }
    except Exception as e:
        return {'Response': f'Unexpected error has occured: {e}, ({type(e)})'}, 500

@app.route("/predict", methods=['POST'])  # Изменили на POST
async def predict():
    try:
        # Проверяем, что main_model инициализирован
        if nn_script.main_model is None:
            return jsonify({'error': 'Model not initialized. Run main() first.'}), 500

        # Получаем JSON-данные
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        url = data.get('url')
        if not url:
            return jsonify({'error': 'URL cannot be empty'}), 400

        print(f"url: {url}")

        # Асинхронный запрос к dataset_manager
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://dataset_manager:5000/image_to_inputs?url={quote(url, safe='')}") as image_response:
                image_response.raise_for_status()
                image_data = await image_response.json()
                if 'error' in image_data:
                    return jsonify({'error': image_data['error']}), 400
                image = np.array(image_data['image'])

        # Предсказание
        prediction = nn_script.main_model.predict(np.array([image]), verbose=1)
        prediction_list = prediction.tolist()[0]
        predicted_class = int(np.argmax(prediction_list))  # Упрощённый способ найти класс

        response = {
            'prediction': prediction_list,
            'prediction_animal': 'dog' if predicted_class == 0 else 'cat',
            'predicted_class': predicted_class,
        }
        return jsonify({'response': response}), 200

    except aiohttp.ClientResponseError as e:
        return jsonify({'error': f"Failed to fetch image: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    from hypercorn.config import Config
    from hypercorn.asyncio import serve
    config = Config()
    config.bind = ["0.0.0.0:5001"]
    asyncio.run(main())
    # go_epochs(5)
    asyncio.run(serve(app, config))