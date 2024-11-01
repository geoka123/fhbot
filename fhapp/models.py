from django.db import models
from django.contrib.auth.models import AbstractUser

import random

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from qdrant_client import QdrantClient

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex

# Create your models here.

class Bot(models.Model):
    botName = models.CharField(max_length=100, unique=True)
    botAPIkey = models.CharField(max_length=250)
    botId = models.IntegerField(null=True, blank=True)
    botLLMModel = models.CharField(max_length=100)

    def get_bot_id(self):
        return self.botId

    def get_bot_name(self):
        return str(self.botName)
    
    def get_bot_api(self):
        return str(self.botAPIkey)
    
    def get_bot_llm(self):
        return str(self.botLLMModel)

    def save(self, *args, **kwargs):
        if not self.pk:
            while True:
                rand_val = random.randint(500, 11000)

                if not Bot.objects.filter(botId=rand_val).exists():
                    self.botId = rand_val
                    break
                else:
                    raise ValueError("Bot with such id already exists")
        super().save(*args, **kwargs)

    def __str__(self):
        return str(self.botName)
    
class User(AbstractUser):
    connectedBot = models.ForeignKey(Bot, on_delete=models.SET_NULL, null=True, blank=True)

class DataSource(models.Model):
    bot = models.ForeignKey(Bot, on_delete=models.CASCADE, related_name="data_sources")
    file = models.FileField(upload_to="data_sources/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.bot.botName} - Data Source ({self.uploaded_at.date()})"


class QuestionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="question_history")
    bot = models.ForeignKey(Bot, on_delete=models.CASCADE, related_name="question_history")
    question_text = models.TextField()
    response_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    visualization_type = models.CharField(max_length=50, blank=True, null=True)  # e.g., 'chart', 'table'
    data_used = models.ForeignKey(DataSource, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"Q: {self.question_text[:50]}... - Bot: {self.bot.botName}"