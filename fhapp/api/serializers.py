from rest_framework import serializers
from ..models import Bot

class BotModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bot
        fields = '__all__'