import os
from pyexpat.errors import messages
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
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver # For automatic memory!
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


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

model = None
app = None

def graph_init():
    global app
    workflow = StateGraph(AgentState)
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
    tool_node = ToolNode(tools)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

def call_model(state: AgentState):
    global model
    response = model.invoke(state["messages"])
    return {"messages": [response]}



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
    model = llm.bind_tools(tools)


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
    model = llm.bind_tools(tools)


def model_init_api():
    global model
    os.environ["OPENAI_API_KEY"] = settings.OPEN_AI_KEY
    llm = ChatOpenAI(
        model="model",
        openai_api_base=settings.MODEL_API_URL,
        openai_api_key="sk-no-key-required",
    )
    if settings.HAS_WEAVAITEDB== "True":
        tools = [
            findBigger,
            sortList_asc,
            sortList_desc,
            is_prime,
            find_factors,
            genRandomNumber,
            write_file,
            query_vector,
            sumNumbers,
        ]
    else:
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
    model = llm.bind_tools(tools)


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
    model = chat_model.bind_tools(tools)


def model_predict(input_text, userid, conversessionID):
    global app
    config = {"configurable": {"thread_id": f"{userid}_{conversessionID}"}}
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
    output = app.invoke(
            {"messages": input_message.messages},
            config=config
    )
    last_message = output["messages"][-1].content
    if "<think>" in last_message and "</think>" in last_message:
        last_message = remove_think(last_message)
    set_memory({"role": "assistant", "content": last_message}, userid, conversessionID)
    return last_message, conversessionID


def model_predict_retry(userid, conversessionID):
    global app
    config = {"configurable": {"thread_id": f"{userid}_{conversessionID}"}}
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
        output = app.invoke(
            {"messages": input_message.messages},
            config=config
        )
    except Exception as e:
        print(e)
        raise
    result = output["messages"][-1].content
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
    global app
    config = {"configurable": {"thread_id": f"{userid}_{conversessionID}"}}
    messages = get_memory(userid, conversessionID)
    readDoc = queryVector(conversessionID, f"{input_text}")

    if messages is None:
        set_memory(
            {
                "role": "assistant",
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

    ragPersonal = f"{settings.SYSTEM_PROMPTS['Default_Personal']}" + readDoc
    messages = [
        {
            "role": "assistant",
            "content": f"{ragPersonal},This is current chat_id:{conversessionID},use it to query vector store with message if you need it",
        },
        {"role": "user", "content": input_text},
    ]
    input_message = load_history_from_json(memory_to_turple(messages))
    output = app.invoke(
        {"messages": input_message.messages},
        config=config
    )
    result = output["messages"][-1].content
    if "<think>" in result and "</think>" in result:
        result = remove_think(result)
    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
    return result, conversessionID


def rag_predict_openai(input_text, userid, conversessionID):
    global app
    config = {"configurable": {"thread_id": f"{userid}_{conversessionID}"}}
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
    output = app.invoke(
        {"messages": input_message.messages},
        config=config
    )
    result = output["messages"][-1].content

    set_memory({"role": "assistant", "content": result}, userid, conversessionID)
    return result, conversessionID


def rag_predict_retry(userid, conversessionID):
    global app
    config = {"configurable": {"thread_id": f"{userid}_{conversessionID}"}}
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
        output = app.invoke(
            {"messages": input_message.messages},
            config=config
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
    global app
    config = {"configurable": {"thread_id": f"{userid}_{conversessionID}"}}
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
        output = app.invoke(
            {"messages": input_message.messages},
            config=config
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



