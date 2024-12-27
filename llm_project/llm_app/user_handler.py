import os,json
from django.conf import settings
import hashlib


user_record_path=settings.USER_DATA_FILE
blank_sha="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

def get_all_user():
    with open(user_record_path,'r') as f:
        users = json.load(f)
    # Exclude `user_ps` from the data
    user_list = [
        {"user_acc": user["user_acc"], "role": user["role"]} for user in users
    ]
    return user_list

def encode_string(string_need_sha):
    shaed_str = hashlib.sha256(string_need_sha.encode()).hexdigest()
    return shaed_str


def add_user(user_acc, user_ps, role=""):
    """Add a new user to the user record."""
    with open(user_record_path, 'r') as f:
        users = json.load(f)

    # Check if the user already exists
    if any(user["user_acc"] == user_acc for user in users):
        return False

    # Add the new user
    users.append({"user_acc": user_acc, "user_ps": encode_string(user_ps), "role": role})

    # Write back to the file
    with open(user_record_path, 'w') as f:
        json.dump(users, f, indent=4)

    return True

def edit_user(ori_user_acc,new_user_acc, user_ps, role=""):
    """edit a user to the user record."""
    with open(user_record_path, 'r') as f:
        users = json.load(f)

    data_dict = {user["user_acc"]: user for user in users}
    if ori_user_acc in data_dict:
        if new_user_acc!= None or new_user_acc!='':
            data_dict[ori_user_acc]["user_acc"] = new_user_acc
        if user_ps!= None and user_ps!='':
            data_dict[ori_user_acc]["user_ps"] = encode_string(user_ps)
        data_dict[ori_user_acc]["role"] = role
        updated_data = list(data_dict.values())
        # Write back to the file
        with open(user_record_path, 'w') as f:
            json.dump(updated_data, f, indent=4)

        return True
    else:
        False


def query_user(user_acc):
    """Query user information by account name."""
    with open(user_record_path, 'r') as f:
        users = json.load(f)

    for user in users:
        if user["user_acc"] == user_acc:
            # Return user details except for the password
            return {"user_acc": user["user_acc"], "role": user["role"]}

    return None




def del_user(user_acc):
    """Delete a user by account name."""
    with open(user_record_path, 'r') as f:
        users = json.load(f)

    # Find the user to delete
    updated_users = [user for user in users if user["user_acc"] != user_acc]

    if len(updated_users) == len(users):
        return False

    # Write back to the file
    with open(user_record_path, 'w') as f:
        json.dump(updated_users, f, indent=4)

    return True
