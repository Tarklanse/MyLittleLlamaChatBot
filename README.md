===LittleLlamaChatBot===  
這是一個以Django框架寫成的聊天機器人主持小專案，可以主持GGUF格式的LLM  
並且有提供簡單的登入介面  
若是在本地端有安裝weaviate的Docker，可以使用簡單的RAG功能  

使用方法:  
需先行安裝requirements.txt中的函式庫
你需要先在llm_project/llm_app/models/這個路徑中放入要執行的GGUF模型，並更新再settings.py中MODEL_PATH指定的模型檔名
若在本地有安裝weaviate，在settings.py中將HAS_WEAVAITEDB改為True後畫面會額外顯示接受PDF的上傳

設定:  
帳號密碼在llm_project/llm_app/user.json中  
在llm_project/llm_project/settings.py中的SYSTEM_PROMPTS中有預設的系統提示詞，welcome_message是所有對話的系統提示詞
