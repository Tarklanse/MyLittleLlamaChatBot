# llm_app/auth_backend.py
import json
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth.models import AbstractBaseUser

class SimpleUser(AbstractBaseUser):
    def __init__(self, username, password=None):
        self.username = username
        self.password = password
        self.backend=""
        self.is_authenticated = True

    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    @property
    def is_staff(self):
        return False

    def get_username(self):
        return self.username

    def get_session_auth_hash(self):
        # Provide a fake session auth hash for the session system
        return make_password(self.password)

    @property
    def pk(self):
        return self.username  # Use the username as the primary key


class JsonFileBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        # Load the user data from the JSON file
        with open('llm_project/llm_app/users.json','r') as f:
            users = json.load(f)

        # Check if the username and password match any entry in the JSON file
        for user in users:
            if user['user_acc'] == username and check_password(password, make_password(user['user_ps'])):
                return SimpleUser(username, password=user['user_ps'])  # Return the simple user object

        return None

    def get_user(self, user_id):
        # We don't need to fetch users from the database, return None
        return None
