import os
import json
from datetime import datetime

# Anchored to this file's location so the path is correct regardless of CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_vectorMapper_PATH = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "vectorMapper"))
Base_Record_filename= "VectorMapper"

def get_vectorMapper_path():
    """Return path for user's memory folder."""
    return os.path.join(BASE_vectorMapper_PATH,f"{Base_Record_filename}.json")
def get_vectorMapper_dir_path():
    """Return path for user's memory folder."""
    return os.path.join(BASE_vectorMapper_PATH)

def add_mapping(chatid,vectorid):
    theFilePath=get_vectorMapper_path()
    # Check if the file exists
    os.makedirs(get_vectorMapper_dir_path(), exist_ok=True)
    if os.path.exists(theFilePath):
        # Load existing chat history
        with open(theFilePath, 'r') as file:
            vecter_mapping = json.load(file)
    else:
        # Start a new chat history
        vecter_mapping = {}

    # Append the new message
    vecter_mapping[chatid] = vectorid

    # Write the updated chat history to the file
    with open(theFilePath, 'w') as file:
        json.dump(vecter_mapping, file)

def get_mapping(chat_id):
    filepath = get_vectorMapper_path()
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data.get(chat_id, None)

def get_All_mapping():
    filepath = get_vectorMapper_path()
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data
    else:
        return {}

def remove_mapping(chat_id):
    filepath = get_vectorMapper_path()
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        if chat_id in data:
            del data[chat_id]
            with open(filepath, 'w') as file:
                json.dump(data, file)
            return True
    return False

def clear_all_mappings():
    filepath = get_vectorMapper_path()
    os.makedirs(get_vectorMapper_dir_path(), exist_ok=True)
    with open(filepath, 'w') as file:
        json.dump({}, file)
