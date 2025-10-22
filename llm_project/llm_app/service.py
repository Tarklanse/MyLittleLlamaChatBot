import os
from django.conf import settings
from .memory_handler import (
    set_memory,
    revert_memory,
    get_memory,
    memory_to_turple,
    load_history_from_json,
)
from .weaviateVectorStoreHandler import queryVector
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.tools import tool
from langchain_core.messages.base import BaseMessage
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import TypedDict, Annotated

import re


from .llmTools import (
    findBigger,
    sortList_asc,
    sortList_desc,
    is_prime,
    find_factors,
    genRandomNumber,
    write_file,
    query_vector,
    custom_code,
    sumNumbers,
)

model = None


def model_init_gguf():
    global model
    model_path = settings.MODEL_PATH
    if settings.MODEL_CHAT_FORMAT == "":
        llm = ChatLlamaCpp(
            model_path=model_path,
            model_kwargs={"tensorcores": True},
            n_ctx=8192,
            n_gpu_layers=-1,
            top_k=0,
            top_p=1,
            max_tokens=settings.GEN_MAX_TOKEN,
            temperature=settings.GEN_TEMPERATURE,
            repeat_penalty=settings.GEN_REPEAT_PENALTY,
        )
    else:
        llm = ChatLlamaCpp(
            model_path=model_path,
            model_kwargs={
                "chat_format": settings.MODEL_CHAT_FORMAT,
                "tensorcores": True,
            },
            n_ctx=8192,
            n_gpu_layers=-1,
            top_k=0,
            top_p=1,
            max_tokens=settings.GEN_MAX_TOKEN,
            temperature=settings.GEN_TEMPERATURE,
            repeat_penalty=settings.GEN_REPEAT_PENALTY,
        )
    tools = [
        findBigger,
        sortList_asc,
        sortList_desc,
        is_prime,
        find_factors,
        genRandomNumber,
        write_file,
        sumNumbers,
    ]
    model = create_agent(
        model=llm,
        tools=tools,
        state_schema=CustomState,
        debug=True,
        prompt=settings.SYSTEM_PROMPTS["Default_Personal"],
    )


def model_init_opanai():
    global model
    os.environ["OPENAI_API_KEY"] = settings.OPEN_AI_KEY
    llm = ChatOpenAI(model="gpt4o")
    tools = [
        findBigger,
        sortList_asc,
        sortList_desc,
        is_prime,
        find_factors,
        genRandomNumber,
        write_file,
        custom_code,
        query_vector,
        sumNumbers,
    ]
    model = create_agent(
        model=llm,
        tools=tools,
        state_schema=CustomState,
        debug=True,
        system_prompt=settings.SYSTEM_PROMPTS["Default_Personal"],
    )


def model_init_api():
    global model
    os.environ["OPENAI_API_KEY"] = settings.OPEN_AI_KEY
    llm = ChatOpenAI(
        model="model",
        openai_api_base=settings.MODEL_API_URL,
        openai_api_key="sk-no-key-required",
    )
    tools = [
        findBigger,
        sortList_asc,
        sortList_desc,
        is_prime,
        find_factors,
        genRandomNumber,
        write_file,
        # custom_code,
        # query_vector,
        sumNumbers,
    ]
    model = create_agent(
        model=llm,
        tools=tools,
        state_schema=CustomState,
        debug=True,
        system_prompt=settings.SYSTEM_PROMPTS["Default_Personal"],
    )


def model_init_transformer():
    global model
    model_id = settings.MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=settings.GEN_MAX_TOKEN,
        return_full_text=False,
    )
    hf = HuggingFacePipeline(pipeline=pipe)
    chat_model = ChatHuggingFace(llm=hf)
    tools = [
        findBigger,
        sortList_asc,
        sortList_desc,
        is_prime,
        find_factors,
        genRandomNumber,
        write_file,
        sumNumbers,
    ]
    model = create_agent(
        model=chat_model,
        tools=tools,
        state_schema=CustomState,
        debug=True,
        system_prompt=settings.SYSTEM_PROMPTS["Default_Personal"],
    )


def model_predict(input_text, userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        conversessionID = set_memory(
            {
                "role": "system",
                "content": f"{settings.SYSTEM_PROMPTS['Default_Personal']}",
            },
            userid,
            conversessionID,
        )
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)
    else:
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)
    input_message = load_history_from_json(memory_to_turple(messages))
    output = model.invoke(
        {"messages": input_message.messages},
    )
    result = output["messages"][-1].content
    # print(result)
    if "<think>" in result and "</think>" in result:
        result = remove_think(result)

    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
    return result, conversessionID


def model_predict_retry(userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        return True
    else:
        messages = get_memory(userid, conversessionID)
        if messages[-1]["role"] != "user" and messages[-1]["role"] != "system":
            messages = get_memory(userid, conversessionID)[:-1]
            revert_memory(userid, conversessionID)

    try:
        input_message = load_history_from_json(memory_to_turple(messages))
        output = model.invoke(
            {"messages": input_message.messages},
        )
    except Exception as e:
        print(e)
        raise
    result = output["messages"][-1].content
    # print(result)
    if "<think>" in result and "</think>" in result:
        result = remove_think(result)
    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
    return result, conversessionID


def print_stream(graph, inputs, config):
    for s in graph.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


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
    readDoc = queryVector(conversessionID, f"{input_text}")

    if messages is None:
        set_memory(
            {
                "role": "system",
                "content": f"{settings.SYSTEM_PROMPTS['Default_Personal']},This is current chat_id:{conversessionID},use it to query vector store with message if you need it",
            },
            userid,
            conversessionID,
        )
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)
    else:
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)
    # 向量資料庫有查到東西時，使用組合的提示詞

    ragPersonal = f"{settings.SYSTEM_PROMPTS['Default_Personal']}" + readDoc
    # print(ragPersonal)
    messages = [
        {
            "role": "assistant",
            "content": f"{ragPersonal},This is current chat_id:{conversessionID},use it to query vector store with message if you need it",
        },
        {"role": "user", "content": input_text},
    ]
    input_message = load_history_from_json(memory_to_turple(messages))
    output = model.invoke(
        {"messages": input_message.messages},
    )
    result = output["messages"][-1].content
    if "<think>" in result and "</think>" in result:
        result = remove_think(result)
    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
    return result, conversessionID


def rag_predict_openai(input_text, userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None:
        set_memory(
            {
                "role": "system",
                "content": f"{settings.SYSTEM_PROMPTS['Default_Personal']},This is current chat_id:{conversessionID},use it to query vector store with message if you need it",
            },
            userid,
            conversessionID,
        )
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)
    else:
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)

    input_message = load_history_from_json(memory_to_turple(messages))
    output = model.invoke(
        {"messages": input_message.messages},
    )
    result = output["messages"][-1].content

    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
    return result, conversessionID


def rag_predict_retry(userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        return True
    else:
        messages = get_memory(userid, conversessionID)[:-1]

    if messages[-1]["role"] != "user" and messages[-1]["role"] != "system":
        messages = get_memory(userid, conversessionID)[:-1]
        revert_memory(userid, conversessionID)

    last_input = messages[len(messages) - 1]["content"]
    readDoc = queryVector(conversessionID, f"{last_input}")
    ragPersonal = f"{settings.SYSTEM_PROMPTS['Default_Personal']}" + readDoc
    # print(ragPersonal)
    messages = [
        {"role": "assistant", "content": f"{ragPersonal}"},
        {"role": "user", "content": last_input},
    ]
    try:
        input_message = load_history_from_json(memory_to_turple(messages))
        output = model.invoke(
            {"messages": input_message.messages},
        )
    except Exception as e:
        print(e)
        raise
    result = output["messages"][-1].content
    if "<think>" in result and "</think>" in result:
        result = remove_think(result)
    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
    return result, conversessionID


def rag_predict_retry_openai(userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        return True
    else:
        messages = get_memory(userid, conversessionID)[:-1]
    if messages[-1]["role"] != "user" and messages[-1]["role"] != "system":
        messages = get_memory(userid, conversessionID)[:-1]
        revert_memory(userid, conversessionID)
    # last_input = messages[len(messages) - 1]["content"]
    try:
        input_message = load_history_from_json(memory_to_turple(messages))
        output = model.invoke(
            {"messages": input_message.messages},
        )
    except Exception as e:
        print(e)
        raise
    result = output["messages"][-1].content
    print(result)
    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
    return result, conversessionID


def remove_think(message):
    return re.sub(r"<think>[\s\S]*?</think>", "", str(message))


class CustomState(TypedDict):
    today: str
    messages: Annotated[list[BaseMessage], add_messages]
    is_last_step: str
    remaining_steps: int
