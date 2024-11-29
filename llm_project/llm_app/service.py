import os
import re
import llama_cpp
import requests
from django.conf import settings
from .memory_handler import set_memory, revert_memory, get_memory, memory_to_turple
from .weaviateVectorStoreHandler import queryVector
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from .llmTools import findBigger,sortList
from langchain_core.messages.base import BaseMessageChunk



memorylib = {}
model = None

def model_init():
    global model
    model_path = settings.MODEL_PATH
    model = ChatLlamaCpp(
        model_path=model_path,
        model_kwargs={"chat_format":settings.MODEL_CHAT_FORMAT,"tensorcores":True},
        n_ctx=8192
    )
    model = create_react_agent(model, tools=[findBigger,sortList])


def model_predict(input_text, userid, conversessionID):
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
        {"messages":memory_to_turple(messages)},
    )
    result = output["messages"][-1].content
    print(result)
    set_memory({"role": "assistant", "content":result}, userid, conversessionID)
    return result, conversessionID

def model_predict_retry(userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        return True
    else:
        messages = get_memory(userid, conversessionID)[:-1]
        revert_memory(userid, conversessionID)

    output = model.invoke(
        {"messages":memory_to_turple(messages)},
    )
    result = output["messages"][-1].content
    print(result)
    set_memory({"role": "assistant", "content":result}, userid, conversessionID)
    return result, conversessionID


def message_undo(userid, conversessionID):
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
        {"messages":memory_to_turple(messages)},
    )
    result = output["messages"][-1].content
    print(result)
    set_memory({"role": "assistant", "content":result}, userid, conversessionID)
    return result, conversessionID

def rag_predict_retry(userid, conversessionID):
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
        {"messages":memory_to_turple(messages)},
    )
    result = output["messages"][-1].content
    print(result)
    set_memory({"role": "assistant", "content":result}, userid, conversessionID)
    return result, conversessionID
