from django.http import JsonResponse, StreamingHttpResponse
from . import utils
import numpy as np

def root(request):
    return JsonResponse(data={'Response': 'This is a root of neural net manager'}, status=200)
