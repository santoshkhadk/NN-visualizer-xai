import json
from django.views.decorators.csrf import csrf_exempt
import base64
import io
from django.http import JsonResponse
from .ml_model import preprocess_canvas_image, predict_top3 ,train_on_sample ,saliency_map,neuron_pixel_contributions,neuron_class_contributions,explain_with_deactivation,integrated_gradients 
import numpy as np



@csrf_exempt
def predict_digit(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            data_url = data.get("image")  

            if not data_url:
                return JsonResponse({"error": "No image provided"}, status=400)

            digit_28x28, X = preprocess_canvas_image(data_url) 

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

            # Preprocess image
            digit_28x28, X = preprocess_canvas_image(img)

            # 1️⃣ Existing deactivation-based explanation
            explanation = explain_with_deactivation(X, deactivate=deactivate, top_k=3)

            # 2️⃣ Integrated Gradients explanation
         # ensure this function is imported
            ig_result = integrated_gradients(X, top_k=3)

# Merge into your explanation dict

           
            # Convert numpy arrays to lists for JSON
            explanation["pixel_heatmap"] = explanation["pixel_heatmap"].tolist()
            explanation["top_neuron_maps"] = [m.tolist() for m in explanation["top_neuron_maps"]]
            explanation["top_neurons"] = explanation["top_neurons"].tolist()
            explanation["hidden_activations"] = explanation["hidden_activations"].tolist()
            explanation["output_probs"] = explanation["output_probs"].tolist()
            explanation["predicted_class"] = int(explanation["predicted_class"])

            explanation["pixel_ig"] = ig_result["pixel_ig"].tolist()
            explanation["neuron_ig"] = ig_result["neuron_ig"].tolist()
            explanation["top_neurons_ig_idx"] = ig_result["top_neurons_idx"].tolist()
            explanation["top_neurons_ig_maps"] = [ m.tolist() for m in ig_result["top_neurons_ig_maps"]]
            return JsonResponse({
                "explanation": explanation,
                "processed_image": digit_28x28.tolist(),
            })

        except Exception as e:
            print("Explain Error:", e)
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)