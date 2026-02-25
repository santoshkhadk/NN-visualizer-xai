import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .ml_model import preprocess_canvas_image, predict_top3  # <-- use predict_top3

@csrf_exempt
def predict_digit(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            data_url = data.get("image")  # base64 canvas image

            if not data_url:
                return JsonResponse({"error": "No image provided"}, status=400)

            # Preprocess canvas image
            X = preprocess_canvas_image(data_url)

            # Get top 3 predictions
            top3 = predict_top3(X)

            # Format for JSON response
            response = [{"digit": digit, "probability": round(prob, 2)} for digit, prob in top3]
            print(response)

            return JsonResponse({"predictions": response})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)