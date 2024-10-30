from django.http import JsonResponse
from ..models import *
from .serializers import *

from rest_framework.decorators import api_view
from rest_framework import viewsets, generics
from rest_framework.response import Response
from rest_framework import status

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from django.conf import settings
import tempfile


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
    queryset = Bot.objects.all()
    serializer_class = BotModelSerializer

    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    vector_store = Chroma(embedding_function=embeddings, persist_directory=None)

    @api_view(['POST'])
    def answer_based_on_text_provided(self, request):
        """Receives botId and input and query and returns answer based ONLY on input text provided"""
        data = request.data
        input_text = data['input']
        query = data['query']

        if not input_text or not query:
            return Response({"error": "Both 'input_text' and 'query' are required"}, status=400)
        
        docs = [{"text": input_text}]
        self.vector_store.add_texts(docs)

        llm = OpenAI(api_key=settings.OPENAI_API_KEY)
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        relevant_docs = self.vector_store.similarity_search(query, k=5)

        answer = qa_chain.run(input_documents=relevant_docs, question=query)
        
        return Response({"answer": answer})
