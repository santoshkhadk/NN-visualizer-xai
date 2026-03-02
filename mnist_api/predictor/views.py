import json
from django.views.decorators.csrf import csrf_exempt
import base64
import io
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
            img = data.get("image")
            if not img:
                return JsonResponse({"error": "No image provided"}, status=400)

            # Preprocess input
            X = preprocess_canvas_image(img)

            # Compute pixel heatmap, probs, hidden activations
            pixel_heatmap, probs, hidden_activations = saliency_map(X)

            # Normalize pixel heatmap
            pixel_heatmap = pixel_heatmap - pixel_heatmap.min()
            pixel_heatmap = pixel_heatmap / (pixel_heatmap.max() + 1e-8)

            # Get top neuron & predicted class
            top_neuron_idx = int(hidden_activations.argmax())
            predicted_class = int(probs.argmax())

            # Return everything for frontend visualization
            return JsonResponse({
                "heatmap": pixel_heatmap.tolist(),
                "hidden_activations": hidden_activations.tolist(),
                "output_probs": probs.tolist(),
                "top_neuron": top_neuron_idx,
                "predicted_class": predicted_class
            })

        except Exception as e:
            print("Explain Error:", e)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)