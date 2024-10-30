from django.db import models

import random

# Create your models here.

class Bot(models.Model):
    botName = models.CharField(max_length=100, unique=True)
    botAPIkey = models.CharField(max_length=250)
    botId = models.IntegerField(null=True, blank=True)

    def get_bot_id(self):
        return self.botId

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