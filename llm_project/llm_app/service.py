import os
import re
import llama_cpp
import requests
from django.conf import settings
from .memory_handler import set_memory, revert_memory, get_memory, memory_to_turple
from .weaviateVectorStoreHandler import queryVector
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.tools import tool
from langchain_core.messages.base import BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import TypedDict, Annotated

from .llmTools import (
    findBigger,
    sortList_asc,
    sortList_desc,
    is_prime,
    find_factors,
    genRandomNumber,
)

memorylib = {}
model = None
graph_Config = None


def model_init_gguf():
    global model, graph_Config
    model_path = settings.MODEL_PATH
    llm = ChatLlamaCpp(
        model_path=model_path,
        model_kwargs={"chat_format": settings.MODEL_CHAT_FORMAT, "tensorcores": True},
        n_ctx=8192,
        max_tokens=settings.GEN_MAX_TOKEN,
        temperature=settings.GEN_TEMPERATURE,
        repeat_penalty=settings.GEN_REPEAT_PENALTY,
    )
    graph_Config = {"configurable": {"thread_id": "thread-1"}}
    tools = [
        findBigger,
        sortList_asc,
        sortList_desc,
        is_prime,
        find_factors,
        genRandomNumber,
    ]
    model = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=CustomState,
        debug=True,
        state_modifier=settings.SYSTEM_PROMPTS["welcome_message"],
    )


def model_init_opanai():
    global model, graph_Config
    os.environ["OPENAI_API_KEY"] = settings.OPEN_AI_KEY
    llm = ChatOpenAI(model="gpt-4")
    graph_Config = {"configurable": {"thread_id": "thread-1"}}
    tools = [
        findBigger,
        sortList_asc,
        sortList_desc,
        is_prime,
        find_factors,
        genRandomNumber,
    ]
    model = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=CustomState,
        debug=True,
        state_modifier=settings.SYSTEM_PROMPTS["welcome_message"],
    )


def model_init_transformer():
    global model, graph_Config
    model_id = settings.MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=settings.GEN_MAX_TOKEN,
    )
    hf = HuggingFacePipeline(pipeline=pipe)
    chat_model = ChatHuggingFace(llm=hf)
    graph_Config = {"configurable": {"thread_id": "thread-1"}}
    tools = [
        findBigger,
        sortList_asc,
        sortList_desc,
        is_prime,
        find_factors,
        genRandomNumber,
    ]
    model = create_react_agent(
        model=chat_model,
        tools=tools,
        state_schema=CustomState,
        debug=True,
        state_modifier=settings.SYSTEM_PROMPTS["welcome_message"],
    )


def model_predict(input_text, userid, conversessionID):
    global model
    global graph_Config
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        # conversessionID = set_memory(
        #     {
        #         "role": "system",
        #         "content": f"{settings.SYSTEM_PROMPTS['welcome_message']}",
        #     },
        #     userid,conversessionID
        # )
        conversessionID = set_memory(
            {"role": "user", "content": input_text}, userid, conversessionID
        )
        messages = get_memory(userid, conversessionID)
    else:
        set_memory({"role": "user", "content": input_text}, userid, conversessionID)
        messages = get_memory(userid, conversessionID)
    output = model.invoke(
        {"messages": memory_to_turple(messages), "is_last_step": False},
    )
    result = output["messages"][-1].content
    print(result)
    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
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
        {"messages": memory_to_turple(messages), "is_last_step": False},
    )
    result = output["messages"][-1].content
    # print(result)
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
                "content": f"{settings.SYSTEM_PROMPTS['welcome_message']}",
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

    ragPersonal = f"{settings.SYSTEM_PROMPTS['welcome_message']}" + readDoc
    # print(ragPersonal)
    messages = [
        {"role": "assistant", "content": f"{ragPersonal}"},
        {"role": "user", "content": input_text},
    ]
    output = model.invoke(
        {"messages": memory_to_turple(messages)},
    )
    result = output["messages"][-1].content
    print(result)
    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
    return result, conversessionID


def rag_predict_retry(userid, conversessionID):
    global model
    messages = get_memory(userid, conversessionID)

    if messages is None or len(messages) == 0:
        return True
    else:
        messages = get_memory(userid, conversessionID)[:-1]
        revert_memory(userid, conversessionID)
    last_input = messages[len(messages) - 1]["content"]
    readDoc = queryVector(conversessionID, f"{last_input}")
    ragPersonal = f"{settings.SYSTEM_PROMPTS['welcome_message']}" + readDoc
    # print(ragPersonal)
    messages = [
        {"role": "assistant", "content": f"{ragPersonal}"},
        {"role": "user", "content": last_input},
    ]
    output = model.invoke(
        {"messages": memory_to_turple(messages)},
    )
    result = output["messages"][-1].content
    print(result)
    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
    return result, conversessionID


class State(TypedDict):
    messages: Annotated[list, add_messages]


class CustomState(TypedDict):
    today: str
    messages: Annotated[list[BaseMessage], add_messages]
    is_last_step: str
