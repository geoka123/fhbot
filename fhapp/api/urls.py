from .views import *
from django.urls import path

urlpatterns = [
    path('hello/', IndexExampleView.index_hello),
    path('botsnow/', AllBotsView.get_all_bots),
    path('uploadfile/', FileUploadView.upload_file_to_data_source),
    path('answerontext/', RespondBasedOnTextProvided.answer_based_on_text_provided, name="bot-query-text"),
]