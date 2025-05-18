import numpy as np
import requests
from urllib3.exceptions import MaxRetryError
import time

print('Neural net script started!')

for i in range(15):
    print(f'Attempt {i}')
    try:
        images_response = requests.get('http://0.0.0.0:5000/get_images')
        images_response.raise_for_status()
        images_json = images_response.json()

        print(images_json)
        print(type(images_json))

        outputs_response = requests.get('http://0.0.0.0:5000/get_outputs')
        outputs_response.raise_for_status()
        outputs_json = outputs_response.json()

        print(outputs_json)
        print(type(outputs_json))
        print('Dataset was got')
        break
    except MaxRetryError as e:
        print('Couldnt connect. Retry in 5 seconds...')
    except Exception as e:
        print(f'Unexpected error has occured: {e}')
    time.sleep(5)