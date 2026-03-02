import json
from django.views.decorators.csrf import csrf_exempt
import base64
import io
from django.http import JsonResponse
from .ml_model import preprocess_canvas_image, predict_top3 ,train_on_sample ,saliency_map,neuron_pixel_contributions,neuron_class_contributions
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

            X = preprocess_canvas_image(img)

            # Call your neuron_class_contributions function which returns everything needed
            top_k = 3
            top_neuron_maps, top_neurons, probs, hidden_activations, predicted_class, neuron_class_contribs = neuron_class_contributions(X, top_k)

            # Pixel saliency map
            pixel_heatmap, _, _ = saliency_map(X)

            # Normalize pixel heatmaps and neuron maps for frontend
            pixel_heatmap = (pixel_heatmap - np.min(pixel_heatmap)) / (np.max(pixel_heatmap) + 1e-8)
            norm_neuron_maps = [(m - np.min(m)) / (np.max(m) + 1e-8) for m in top_neuron_maps]

            return JsonResponse({
                "pixel_heatmap": pixel_heatmap.tolist(),
                "top_neuron_maps": [m.tolist() for m in norm_neuron_maps],
                "top_neurons": top_neurons.tolist(),
                "hidden_activations": hidden_activations.tolist(),
                "output_probs": probs.tolist(),
                "predicted_class": predicted_class,
                "neuron_class_contribs": neuron_class_contribs
            })

        except Exception as e:
            print("Explain Error:", e)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)