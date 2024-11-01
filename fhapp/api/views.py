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

from rest_framework.renderers import JSONRenderer
from rest_framework.decorators import renderer_classes

from django.conf import settings
from langchain_experimental.agents import create_csv_agent
import openpyxl
from langchain.document_loaders.csv_loader import CSVLoader

import tempfile
import logging
import pandas as pd
from langchain_openai import OpenAI

from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from qdrant_client import QdrantClient

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex

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
            parser = LlamaParse(
                api_key=('llx-MHkv4e22IpbbRBvnKRaPZTWXbuVQpozhfmypYJrpSTyEBjcJ'),
                parsing_instruction = f"""You are an agent that takes an excel file that contains the application data of startups to an accelerator program. Each row corresponds to an application. Give detailed answers based on these applications.""",
                result_type="markdown"
            )

            file_extractor = {".xlsx": parser}
            documents = SimpleDirectoryReader(input_files=['/home/ec2-user/fhbot/media/data_sources/Application_Database.xlsx'], file_extractor=file_extractor).load_data()

            llm = Groq(model="llama3-70b-8192", api_key='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')
            Settings.llm = llm


            embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
            Settings.embed_model = embed_model

            qdrant_client = QdrantClient(
                url="https://bff3be45-6a0e-4931-83be-d93c2810171d.us-east4-0.gcp.cloud.qdrant.io:6333", 
                api_key="Pyq_lqp0G9xhTIWiwidSv3evxN98jix72qUjBnFPP8VNKiClwKsTIw",
            )

            vector_store = QdrantVectorStore(client=qdrant_client, collection_name="fh_data")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Create vector store index and store it in Qdrant DB
            VectorStoreIndex.from_documents(documents, storage_context=storage_context)
            serializer.save()
            return Response({"success": "Yes"}, status=status.HTTP_201_CREATED)
        return Response({"success": "No"}, status=status.HTTP_400_BAD_REQUEST)

class RespondBasedOnTextProvided(viewsets.ModelViewSet):
    @api_view(['POST'])
    @renderer_classes([JSONRenderer])
    def answer_based_on_text_provided(request):
        """Receives input and query and returns answer based ONLY on input text provided."""
        data = request.data
        question = data.get('input')
        context_file = str(data.get('file'))
        context = ""
        output_csv = r'/home/ec2-user/fhbot/cur_csvfile.csv'
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=1.0, max_new_tokens=2048, huggingfacehub_api_token="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")

        if not question:
            return Response({"error": "Both 'input' and 'query' are required"}, status=400)
        if context_file == "1":

            embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
            Settings.embed_model = embed_model

            qdrant_client = QdrantClient(
                url="https://bff3be45-6a0e-4931-83be-d93c2810171d.us-east4-0.gcp.cloud.qdrant.io:6333", 
                api_key="Pyq_lqp0G9xhTIWiwidSv3evxN98jix72qUjBnFPP8VNKiClwKsTIw",
                prefer_grpc=True,
            )

            vector_store = QdrantVectorStore(client=qdrant_client, collection_name="fh_data")
            db_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

            # create a query engine for the index
            query_engine = db_index.as_query_engine()
            response = query_engine.query(question)
            return JsonResponse({"text": f"{response}"})


        prompt_template = PromptTemplate(
            input_variables=["question"],
            template="""
            You are a knowledgeable assistant that provides concise, accurate, and well-formatted answers.

            - If code is required, output the full code in Python using matplotlib without extra commentary.
            - If the response is long, continue in a follow-up response to ensure completeness.

            Question: {question}

            Answer:
            """
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)

        input_data = {"question": question}
        result = llm_chain.invoke(input_data)

        return JsonResponse(result)