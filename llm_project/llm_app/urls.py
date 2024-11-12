from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat_view'),
    path('api/llm', views.llm_api, name='llm_api'),
    path('chat/', views.chat_view, name='chat_view'),
    path('chat/retry', views.chat_retry, name='chat_retry'),
    path('chat/undo', views.chat_undo, name='chat_undo'),
    path('chat/reset', views.reset_chat_history, name='chat_reset'),
    path('chat/gethistory', views.get_all_chat_history, name='chat_gethistory'),
    path('chat/loadchat', views.load_chat_history, name='chat_loadchat'),
    path('login/', views.login_view, name='login'),
    path('chat/upload_pdf', views.upload_pdf, name='upload_pdf'),

]
