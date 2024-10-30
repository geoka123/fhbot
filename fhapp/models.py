from django.db import models

import random

# Create your models here.

class Bot(models.Model):
    botName = models.CharField(max_length=100)
    botAPIkey = models.CharField(max_length=250)
    botId = models.IntegerField(null=True, blank=True)

    def get_bot_id(self):
        return self.botId

    def save(self, *args, **kwargs):
        if self.botId == None:
            while True:
                bid = random.randint(1, 10000)
                id_exists = False

                all_bots = Bot.objects.all()
                for bot in all_bots:
                    if bot.get_bot_id() == bid:
                        id_exists = True
                        break
                
                if not id_exists:
                    break
                        
        super().save()