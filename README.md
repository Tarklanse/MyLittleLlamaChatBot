===LittleLlamaChatBot===  
這是一個以Django框架寫成的聊天機器人主持小專案，可以主持GGUF,Transformer格式的LLM,或是串接OpenAI API  
並且有提供簡單的登入介面  
![image](https://github.com/Tarklanse/MyLittleLlamaChatBot/blob/main/login_Page.png?raw=true)  
登入後便能開始與AI對話
![image](https://github.com/Tarklanse/MyLittleLlamaChatBot/blob/main/chat_without_PDF.png?raw=true)
若是在本地端有安裝weaviate的Docker，會出現對應的PDF上傳功能
![image](https://github.com/Tarklanse/MyLittleLlamaChatBot/blob/main/chat_with_pdf_record.png?raw=true)
*圖中範例為拿著Omnigen的論文詢問的結果,原始使用的模型(Gemma2-2b)完全不知道Omnigen是甚麼，但搭配PDF後可以觀察到能正確回答Omnigen是甚麼  

使用方法:  
需先行安裝requirements.txt中的函式庫  
你需要先在llm_project/llm_app/models/這個路徑中放入要執行的GGUF模型，並更新在settings.py中MODEL_PATH指定的模型檔名  
如果要接OpenAI的API則需先行設定MODEL_TYPE為openai,並將你的api key寫於OPEN_AI_KEY
若在本地有安裝weaviate，在settings.py中將HAS_WEAVAITEDB改為True後畫面會額外顯示接受PDF的上傳  
執行以下指令應該能將專案跑起來  
```
python llm_project\manage.py runserver 0.0.0.0:8123 --noreload  
```

設定:  
帳號密碼在llm_project/llm_app/user.json中  
在llm_project/llm_project/settings.py中  
SYSTEM_PROMPTS:系統文字，其中Default_Personal是所有對話的系統提示詞  
MODEL_PATH:是指定的模型路徑  
MODEL_CHAT_FORMAT 對應llama.cpp中的chat format,需注意gemma系列的模型不接受system prompts,使用Gemma系列的Model內部程式會將System prompts轉換為一個隱藏訊息  
GEN_TEMPERATURE: 文字生成溫度  
GEN_REPEAT_PENALTY: 文字重複懲罰  
GEN_MAX_TOKEN:最大回應的Token數量  
