from django.urls import path
from .views import predict_digit,correct_digit

urlpatterns = [
    path("predict_digit/", predict_digit),
    path("correct_digit/", correct_digit),
]