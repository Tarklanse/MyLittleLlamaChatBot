import os
import re
import llama_cpp
import requests
from django.conf import settings
from .memory_handler import set_memory, revert_memory, get_memory, delete_memory
from .weaviateVectorStoreHandler import queryVector
from langchain_community.chat_models import ChatLlamaCpp

memorylib = {}
model = None

def model_init():
    global model
    model_path = settings.MODEL_PATH
    model = ChatLlamaCpp(
        model_path=model_path, model_kwargs={"chat_format":"gemma","tensorcores":True}, n_ctx=8192
    )

def model_predict2(input_text, userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        conversessionID = set_memory(
            {
                "role": "system",
                "content": f"{settings.SYSTEM_PROMPTS['welcome_message']}",
            },
            userid,conversessionID
        )
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)
    else:
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)

    output = model.invoke(
        messages
    )
    result = output.content
    set_memory({"role": "assistant", "content":output.content}, userid, conversessionID)
    return result, conversessionID


def model_predict_retry2(userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        return True
    else:
        messages = get_memory(userid, conversessionID)[:-1]
        revert_memory(userid, conversessionID)

    output = model.invoke(
        messages
    )
    result = output.content
    set_memory({"role": "assistant", "content":output.content}, userid, conversessionID)
    return result, conversessionID


def message_undo2(userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)
    try:
        if messages is None:
            return True
        else:
            revert_memory(userid, conversessionID)
            revert_memory(userid, conversessionID)
            return True
    except Exception as e:
        print(e)
        return False



def rag_predict(input_text, userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)
    readDoc = queryVector(conversessionID,f"{input_text}")

    if messages is None:
        set_memory({"role": "system", "content": f"{settings.SYSTEM_PROMPTS['welcome_message']}"}, userid, conversessionID)
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)
    else:
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)
    # 向量資料庫有查到東西時，使用組合的提示詞
    
    ragPersonal = f"{settings.SYSTEM_PROMPTS['welcome_message']}"+readDoc
    # print(ragPersonal)
    messages = [
        {"role": "assistant", "content": f"{ragPersonal}"},
        {"role": "user", "content": input_text},
    ]
    output = model.invoke(
        messages
    )
    result = output.content
    set_memory({"role": "assistant", "content":output.content}, userid, conversessionID)
    return result, conversessionID

def RAG_predict_retry(userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        return True
    else:
        messages = get_memory(userid, conversessionID)[:-1]
        revert_memory(userid, conversessionID)
    last_input=messages[len(messages)-1]["content"]
    readDoc = queryVector(conversessionID,f"{last_input}")
    ragPersonal = f"{settings.SYSTEM_PROMPTS['welcome_message']}"+readDoc
    # print(ragPersonal)
    messages = [
        {"role": "assistant", "content": f"{ragPersonal}"},
        {"role": "user", "content": last_input},
    ]
    output = model.invoke(
        messages
    )
    result = output.content
    set_memory({"role": "assistant", "content":output.content}, userid, conversessionID)
    return result, conversessionID
