from django.db import models

# Create your models here.

class chatHistory(models.Model):
    ChatTitle = models.CharField(max_length=100)
    FileName = models.CharField(max_length=100)
    vectorID = models.CharField(max_length=100)
    owner = models.CharField(max_length=100)