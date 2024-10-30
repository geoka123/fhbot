from .views import *
from django.urls import path

url_patterns = [
    path('hello/', IndexExampleView.index_hello)
]