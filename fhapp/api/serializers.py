from rest_framework import serializers
from ..models import Bot, DataSource

class BotModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bot
        fields = '__all__'

class FileUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSource
        fields = ['file']