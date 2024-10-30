from .views import *
from django.urls import path

urlpatterns = [
    path('hello/', IndexExampleView.index_hello),
    path('botsnow/', AllBotsView.get_all_bots),
]