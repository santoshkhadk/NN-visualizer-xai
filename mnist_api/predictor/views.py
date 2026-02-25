import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .ml_model import predict, preprocess_canvas_image

@csrf_exempt
def predict_digit(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            data_url = data.get("image")  # base64 canvas image

            if not data_url:
                return JsonResponse({"error": "No image provided"}, status=400)

            # Preprocess and predict
            X = preprocess_canvas_image(data_url)
            result = predict(X)

            return JsonResponse({"prediction": result})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)