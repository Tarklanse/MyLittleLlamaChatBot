import os
import json
from datetime import datetime
from django.conf import settings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.messages  import SystemMessage, HumanMessage, AIMessage

# Directory to store chat histories
BASE_MEMORY_PATH = "memory"


def get_user_path(userid):
    """Return path for user's memory folder."""
    return os.path.join(BASE_MEMORY_PATH, str(userid))


def get_filepath(userid, timestamp):
    """Return the filepath for a specific chat session based on the given timestamp."""
    return os.path.join(get_user_path(userid), f"{timestamp}.json")


def set_memory(newmessage, userid, timestamp=None):
    """Append new messages to an existing memory file or create a new one."""
    user_path = get_user_path(userid)

    # Ensure user directory exists
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    # Generate timestamp if not provided
    if timestamp is None or timestamp == "":
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    memory_file = get_filepath(userid, timestamp)

    # Check if the file exists
    if os.path.exists(memory_file):
        # Load existing chat history
        with open(memory_file, "r") as file:
            chat_history = json.load(file)
    else:
        # Start a new chat history
        chat_history = []

    # Append the new message
    chat_history.append(newmessage)

    # Write the updated chat history to the file
    with open(memory_file, "w") as file:
        json.dump(chat_history, file)

    return timestamp


def revert_memory(userid, timestamp):
    """Overwrite the specified message file with a single new message."""
    filepath = get_filepath(userid, timestamp)
    if os.path.exists(filepath):
        # Load existing chat history
        with open(filepath, "r") as file:
            chat_history = json.load(file)
        with open(filepath, "w") as file:
            json.dump(chat_history[:-1], file)
    else:
        print(f"No file found with timestamp {timestamp} for user {userid}")


def get_memory(userid, timestamp):
    """Load all chat messages for a specific session file based on the given timestamp."""
    if timestamp is None:
        return None
    filepath = get_filepath(userid, timestamp)
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return json.load(file)
    else:
        print(f"No file found with timestamp {timestamp} for user {userid}")
        return None

def load_history_from_json(json_data):
    """
    json_data: list of dicts, each has at least 'role' and 'content'
    returns: a ChatMessageHistory instance with messages
    """
    restored_messages = []
    for m in json_data:
        role = m[0]
        content = m[1]
        if role == "system":
            restored_messages.append(SystemMessage(content=content))
        elif role == "user":
            restored_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            restored_messages.append(AIMessage(content=content))
    chat_history = ChatMessageHistory(messages=restored_messages)
    return chat_history

def edit_persenal(userid, timestamp, newpersonal):
    """Load all chat messages for a specific session file based on the given timestamp."""
    # Generate timestamp if not provided
    if timestamp is None or timestamp == "":
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filepath = get_filepath(userid, timestamp)
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            chat_history = json.load(file)
        if chat_history[0]["role"] == "user":
            chat_history = [{"role": "system", "content": newpersonal}].append(
                chat_history
            )
        else:
            chat_history[0]["content"] = newpersonal

        with open(filepath, "w") as file:
            json.dump(chat_history, file)
        return timestamp
    else:
        print(f"No file found with timestamp {timestamp} for user {userid}")
        chat_history = [{"role": "system", "content": newpersonal}]
        with open(filepath, "w") as file:
            json.dump(chat_history, file)
        return timestamp


def delete_memory(userid, timestamp):
    """Delete a specific chat history file based on the given timestamp."""
    filepath = get_filepath(userid, timestamp)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Deleted file with timestamp {timestamp} for user {userid}")
        return True
    else:
        print(f"No file found with timestamp {timestamp} for user {userid}")
        return False


def list_memory_sessions(userid):
    """List all memory session timestamps for a given user."""
    user_path = get_user_path(userid) + "/"

    if not os.path.exists(user_path):
        print(f"No memory folder found for user {userid}")
        return []

    # List all JSON files in the user's directory and extract timestamps
    memory_sessions = []
    for filename in os.listdir(user_path):
        # print(filename)
        if filename.endswith(".json"):
            timestamp = filename.replace(".json", "")
            memory_sessions.append(timestamp)
    return sorted(memory_sessions, reverse=True)


def memory_to_turple(history_list: list):
    result = []
    for obj in history_list:
        if settings.MODEL_CHAT_FORMAT == "gemma" and obj["role"] == "system":
            result.append(("assistant", obj["content"]))
        else:
            result.append((obj["role"], obj["content"]))
    return result
