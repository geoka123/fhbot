from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(Bot)
admin.site.register(User)
admin.site.register(DataSource)
admin.site.register(QuestionHistory)