from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
from rest_framework.views import APIView
from ..models import *
from .serializers import *

from rest_framework.decorators import api_view, action
from rest_framework import viewsets, generics
from rest_framework.response import Response
from rest_framework import status

from django.conf import settings
import tempfile
import logging

from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexExampleView(viewsets.ViewSet):
    @api_view(['GET'])
    def index_hello(request):
        return JsonResponse({"EIMAI MALAKAS": "YES"})
    
class AllBotsView(viewsets.ModelViewSet):
    queryset = Bot.objects.all()
    serializer_class = BotModelSerializer

    @api_view(['GET'])
    def get_all_bots(request):
        counter = 1
        bot_dict = dict()
        all_bots = Bot.objects.all()

        for bot in all_bots:
            bot_dict[counter] = {"botName": bot.get_bot_name(), "botId": bot.get_bot_id(), "botAPIkey": bot.get_bot_api(), "botLLMModel": bot.get_bot_llm()}
            counter += 1
        
        return JsonResponse(bot_dict)

class FileUploadView(viewsets.ModelViewSet):
    queryset = DataSource.objects.all()
    serializer_class = FileUploadSerializer

    @api_view(['POST'])
    def upload_file_to_data_source(request):
        """Accepts a file, returns json with success: yes and status 201 if everything is ok, else success: no and status 400"""
        file_to_upload = request.data
        serializer = FileUploadSerializer(data=file_to_upload)

        if serializer.is_valid():
            serializer.save()
            return Response({"success": "Yes"}, status=status.HTTP_201_CREATED)
        return Response({"success": "No"}, status=status.HTTP_400_BAD_REQUEST)

class RespondBasedOnTextProvided(viewsets.ModelViewSet):
    @api_view(['POST'])
    def answer_based_on_text_provided(request):
        """Receives input and query and returns answer based ONLY on input text provided."""
        data = request.data
        question = data.get('input')

        if not question:
            return Response({"error": "Both 'input' and 'query' are required"}, status=400)

        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=10000, temperature=0.7, token="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")

        prompt_template = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an intelligent assistant. Answer each question in detail.

            Question: {question}

            Answer:
            """
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Attempt to get a non-blank response with a retry mechanism
        complete_answer = ""
        retries = 3
        attempt = 0

        while attempt < retries:
            input_data = {"question": f"{question} {complete_answer}" if complete_answer else question}
            result = llm_chain.invoke(input_data)

            # Check for a blank response and retry if necessary
            partial_answer = result.get("text", "").strip()
            if partial_answer:
                complete_answer += partial_answer
                if "continue" not in partial_answer.lower():
                    break  # Exit loop if response is complete
            else:
                logger.warning("Received a blank response, retrying...")
                attempt += 1
        
        # Final check if no valid response was obtained
        if not complete_answer:
            logger.error("Failed to obtain a non-blank response after retries.")
            return JsonResponse({"error": "The assistant failed to provide a response. Please try again later."}, status=500)

        return JsonResponse({"question": question, "answer": complete_answer})