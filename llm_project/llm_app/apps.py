from django.apps import AppConfig
from django.conf import settings
import os
from .service import model_init_gguf,model_init_opanai,model_init_api,model_init_transformer,graph_init
import queue


class LlmAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'llm_app'

    def ready(self):
        print('ready')
        os.environ['LANGSMITH_TRACING'] = settings.LANGSMITH_TRACING
        os.environ['LANGSMITH_ENDPOINT'] = settings.LANGSMITH_ENDPOINT
        os.environ['LANGSMITH_API_KEY'] = settings.LANGSMITH_API_KEY
        os.environ['LANGSMITH_PROJECT'] = settings.LANGSMITH_PROJECT
        if settings.MODEL_TYPE == 'transformer':
            model_init_transformer()
            graph_init()
        elif settings.MODEL_TYPE == 'openai':
            model_init_opanai()
            graph_init()
        elif settings.MODEL_TYPE == 'api':
            model_init_api()
            graph_init()
        else:
            model_init_gguf()
            graph_init()
