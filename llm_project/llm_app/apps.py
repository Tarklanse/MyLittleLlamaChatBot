from django.apps import AppConfig
from django.conf import settings
from .service import model_init_gguf,model_init_opanai,model_init_api,model_init_transformer
import queue


class LlmAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'llm_app'

    def ready(self):
        print('ready')
        if settings.MODEL_TYPE == 'transformer':
            model_init_transformer()
        elif settings.MODEL_TYPE == 'openai':
            model_init_opanai()
        elif settings.MODEL_TYPE == 'api':
            model_init_api()
        else:
            model_init_gguf()
