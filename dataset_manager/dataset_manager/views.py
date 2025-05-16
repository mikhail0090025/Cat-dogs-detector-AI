from django.http import JsonResponse, StreamingHttpResponse
from . import utils
import numpy as np

def root(request):
    return JsonResponse(data={'Response': 'This is a root of dataset manager'}, status=200)

def get_images(request):
    images = utils.images
    def iterfile():
        yield images.tobytes()
    return StreamingHttpResponse(iterfile(), content_type="application/octet-stream")

def get_outputs(request):
    outputs = utils.outputs
    def iterfile():
        yield outputs.tobytes()
    return StreamingHttpResponse(iterfile(), content_type="application/octet-stream")