import json
from django.views.decorators.csrf import csrf_exempt
import base64
import io
from django.http import JsonResponse
from .ml_model import preprocess_canvas_image, predict_top3 ,train_on_sample ,saliency_map,neuron_pixel_contributions,neuron_class_contributions,explain_with_deactivation
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
            deactivate = data.get("deactivate_neurons", [])

            if not img:
                return JsonResponse({"error": "No image provided"}, status=400)

            X = preprocess_canvas_image(img)

            # Call the model function with deactivation
            explanation = explain_with_deactivation(X, deactivate=deactivate, top_k=3)

            # Recompute prediction after deactivation
            # (explanation["output_probs"] should already be logits/probs from the deactivated network)
            probs_after = explanation["output_probs"]
            if isinstance(probs_after, np.ndarray):
                probs_after = probs_after.flatten()
            pred_class_after = int(np.argmax(probs_after))

            # Convert numpy arrays to lists for JSON
            explanation["pixel_heatmap"] = explanation["pixel_heatmap"].tolist()
            explanation["top_neuron_maps"] = [m.tolist() for m in explanation["top_neuron_maps"]]
            explanation["top_neurons"] = explanation["top_neurons"].tolist()
            explanation["hidden_activations"] = explanation["hidden_activations"].tolist()
            explanation["output_probs"] = probs_after.tolist()
            explanation["predicted_class"] = pred_class_after  # Add updated prediction

            return JsonResponse(explanation)

        except Exception as e:
            print("Explain Error:", e)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)