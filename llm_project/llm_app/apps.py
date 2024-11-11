from django.apps import AppConfig
from django.conf import settings
from .service import model_init
import queue


class LlmAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'llm_app'

    def ready(self):
        model_init()
        
