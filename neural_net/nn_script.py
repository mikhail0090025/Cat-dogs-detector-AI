import numpy as np
import aiohttp
import asyncio
import json

async def fetch_data(session, url):
    async with session.get(url, timeout=10, ssl=False) as response:
        response.raise_for_status()
        data = await response.json()
        return data

async def main():
    print('Neural net script started!', flush=True)

    async with aiohttp.ClientSession() as session:
        for i in range(40):
            print(f'Attempt {i}', flush=True)
            try:
                images_data = await fetch_data(session, 'http://dataset_manager:5000/get_images')
                images = np.array(images_data['images'])
                outputs_data = await fetch_data(session, 'http://dataset_manager:5000/get_outputs')
                outputs = np.array(outputs_data['outputs'])

                print(images)
                print(outputs)

                print('Images shape:', images.shape, flush=True)
                print('Outputs shape:', outputs.shape, flush=True)
                print('Dataset was got', flush=True)
                break
            except aiohttp.ClientError as e:
                print(f'Unexpected error has occurred: {e}', flush=True)
                print('Couldnt connect. Retry in 10 seconds...', flush=True)
                await asyncio.sleep(10)
            except Exception as e:
                print(f'Unexpected error has occurred: {e}', flush=True)
                await asyncio.sleep(10) 

if __name__ == '__main__':
    asyncio.run(main())