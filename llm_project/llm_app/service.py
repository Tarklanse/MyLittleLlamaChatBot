import os
import re
import llama_cpp
import requests
from django.conf import settings
from .memory_handler import set_memory, revert_memory, get_memory, delete_memory
from .weaviateVectorStoreHandler import queryVector

memorylib = {}
model = None


def setmemory(newmessage, userid):
    global memorylib
    if userid in memorylib:
        memorylib[userid].append(newmessage)
    else:
        memorylib[userid] = [newmessage]


def revertmemory(newmessage, userid):
    global memorylib
    if userid in memorylib:
        memorylib[userid] = [newmessage]


def getmemory(userid):
    global memorylib
    if userid in memorylib:
        return memorylib[userid]
    else:
        return None


def delmemory(userid):
    global memorylib
    if userid in memorylib:
        return memorylib.pop(userid)
    else:
        return None


def model_init():
    global model
    model_path = settings.MODEL_PATH
    model = llama_cpp.Llama(
        model_path, chat_format="gemma", n_ctx=8192, n_gpu_layers=26, tensorcores=True
    )


def model_predict(input_text, userid):
    global model
    messages = getmemory(userid)

    if messages is None or len(messages) == 0:
        setmemory(
            {
                "role": "system",
                "content": f"{settings.SYSTEM_PROMPTS['welcome_message']}",
            },
            userid,
        )
        setmemory({"role": "user", "content": input_text}, userid)
        messages = getmemory(userid)
    else:
        setmemory({"role": "user", "content": input_text}, userid)
        messages = getmemory(userid)

    output = model.create_chat_completion(
        messages, max_tokens=4096, temperature=1, repeat_penalty=1.5
    )
    print(output)
    result = output["choices"][0]["message"]["content"]
    setmemory(output["choices"][0]["message"], userid)
    return result


def model_predict_retry(userid):
    global model
    messages = getmemory(userid)

    if messages is None or len(messages) == 0:
        return True
    else:
        messages = getmemory(userid)[:-1]
        revertmemory(messages, userid)

    output = model.create_chat_completion(
        messages, max_tokens=4096, temperature=1, repeat_penalty=1.5
    )
    print(output)
    result = output["choices"][0]["message"]["content"]
    setmemory(output["choices"][0]["message"], userid)
    return result


def message_undo(userid):
    global model
    messages = getmemory(userid)
    try:
        if messages is None:
            return True
        else:
            messages = getmemory(userid)[:-2]
            revertmemory(messages, userid)
            return True
    except Exception as e:
        print(e)
        return False


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

    output = model.create_chat_completion(
        messages, max_tokens=4096, temperature=1, repeat_penalty=1.5
    )
    print(output)
    result = output["choices"][0]["message"]["content"]
    set_memory(output["choices"][0]["message"], userid, conversessionID)
    return result, conversessionID


def model_predict_retry2(userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        return True
    else:
        messages = get_memory(userid, conversessionID)[:-1]
        revert_memory(userid, conversessionID)

    output = model.create_chat_completion(
        messages, max_tokens=4096, temperature=1, repeat_penalty=1.5
    )
    print(output)
    result = output["choices"][0]["message"]["content"]
    set_memory(output["choices"][0]["message"], userid, conversessionID)
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
    output = model.create_chat_completion(
        messages, max_tokens=4096, temperature=0.2, repeat_penalty=1.5
    )
    result = output["choices"][0]["message"]["content"]
    set_memory(output["choices"][0]["message"], userid, conversessionID)
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
    output = model.create_chat_completion(
        messages, max_tokens=4096, temperature=1, repeat_penalty=1.5
    )
    result = output["choices"][0]["message"]["content"]
    set_memory(output["choices"][0]["message"], userid, conversessionID)
    return result, conversessionID
