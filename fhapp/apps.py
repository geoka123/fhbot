from django.apps import AppConfig

class FhappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'fhapp'

class MyAppConfig(AppConfig):
    name = 'init_llm'

    def ready(self):
        # Import and call your custom functions here
        from .llm_rag_config import lllm_init, index_init, query_engine_init
        lllm_init()
        index_init()
        query_engine_init()
