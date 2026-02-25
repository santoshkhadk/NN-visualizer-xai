import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .ml_model import preprocess_canvas_image, predict_top3 ,train_on_sample ,saliency_map
import numpy as np

@csrf_exempt
def predict_digit(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            data_url = data.get("image")  

            if not data_url:
                return JsonResponse({"error": "No image provided"}, status=400)

         
            X = preprocess_canvas_image(data_url)

            top3 = predict_top3(X)

          
            response = [{"digit": digit, "probability": round(prob, 2)} for digit, prob in top3]
            print(response)

            return JsonResponse({"predictions": response})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
@csrf_exempt
def correct_digit(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            data_url = data.get("image")
            label = int(data.get("label"))

            if data_url is None:
                return JsonResponse({"error": "No image"}, status=400)

            X = preprocess_canvas_image(data_url)

            probs = train_on_sample(X, label)

            return JsonResponse({
                "message": "Model trained in real-time",
                "new_prediction": int(probs.argmax())
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
@csrf_exempt
def explain_digit(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            data_url = data.get("image")

            X = preprocess_canvas_image(data_url)

            heatmap, probs = saliency_map(X)
            print(heatmap[0])

            return JsonResponse({
                "heatmap": heatmap.tolist(),
                "prediction": int(np.argmax(probs))
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)