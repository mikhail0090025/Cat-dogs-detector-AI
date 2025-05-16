from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse, HTMLResponse, StreamingResponse
import utils
import numpy as np

app = FastAPI()

app.route('/')
def root():
    return JSONResponse({'Response': 'This is a root of dataset manager'}, status_code=200)

app.get('/get_images')
def get_images():
    images = utils.images
    def iterfile():
        yield images.tobytes()

    return StreamingResponse(iterfile(), media_type="application/octet-stream")

app.get('/get_outputs')
def get_outputs():
    outputs = utils.outputs
    def iterfile():
        yield outputs.tobytes()

    return StreamingResponse(iterfile(), media_type="application/octet-stream")