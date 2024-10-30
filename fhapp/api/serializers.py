from rest_framework import serializers
from ..models import Bot, DataSource

class BotModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bot
        fields = '__all__'

class FileUploadSerializer(serializers.ModelSerializer):
    botId = serializers.IntegerField(write_only=True)

    class Meta:
        model = DataSource
        fields = ['file', 'botId']

    def create(self, validated_data):
        botId = validated_data.pop('botId')
        bot = Bot.objects.get(botId=botId)  # Retrieve the Bot instance
        return DataSource.objects.create(bot=bot, **validated_data)