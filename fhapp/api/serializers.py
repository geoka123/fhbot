from rest_framework import serializers
from ..models import Bot, DataSource

class BotModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bot
        fields = '__all__'

class FileUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSource
        fields = ['file', 'botId']

    def create(self, validated_data):
        bot_id = validated_data.pop('bot_id')
        bot = Bot.objects.get(id=bot_id)  # Retrieve the Bot instance
        return DataSource.objects.create(bot=bot, **validated_data)