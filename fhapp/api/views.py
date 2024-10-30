from django.http import JsonResponse

from rest_framework.decorators import api_view
from rest_framework import viewsets, generics

class IndexExampleView(viewsets.ViewSet):
    @api_view(['GET'])
    def index_hello(request):
        return JsonResponse({"EIMAI MALAKAS": "YES"})