import json
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .ml_model import predict


@csrf_exempt
def predict_digit(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
            pixels = np.array(body["pixels"]).reshape(1, -1)

            result = predict(pixels)

            return JsonResponse({
                "prediction": result
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"message": "Send POST request"})