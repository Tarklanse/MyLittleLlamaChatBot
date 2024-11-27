from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from .service import (
    model_predict_retry,
    model_predict,
    message_undo,
    rag_predict,
    rag_predict_retry,
    model_text_only_predict
)
from .memory_handler import list_memory_sessions, delete_memory, get_memory
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework import status
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .auth_backend import JsonFileBackend
import traceback
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
import os
from datetime import datetime
from .weaviateVectorStoreHandler import newVector
from .vectorMapper import get_mapping


@api_view(["POST"])
def llm_api(request):
    try:
        if request.session.get("is_authenticated"):
            user_id = request.session.get("user_id")
            if "prompt" not in request.data:
                return Response(
                    {"error": "Prompt is required"}, status=status.HTTP_400_BAD_REQUEST
                )

            prompt = request.data.get("prompt")
            tempid = request.data.get("CCID")
            vectorId=get_mapping(tempid)
            if vectorId is None or settings.HAS_WEAVAITEDB is False:
                result, conversessionID = model_predict(prompt, user_id, tempid)
            else:
                result, conversessionID = rag_predict(prompt, user_id, tempid)
            return Response(
                {"result": result, "CCID": conversessionID}, status=status.HTTP_200_OK
            )
        else:
            return Response(
                {"redirect_url": "/login/"}, status=status.HTTP_403_FORBIDDEN
            )  # Return a redirect URL and 403 status

    except Exception as e:
        print(e)
        traceback.print_exc()
        return Response(
            {"result": settings.SYSTEM_PROMPTS["error_message"]},
            status=status.HTTP_200_OK,
        )

@api_view(["POST"])
def chat_retry(request):
    try:
        if request.session.get("is_authenticated"):
            user_id = request.session.get("user_id")
            tempid = request.data.get("CCID")
            vectorId=get_mapping(tempid)
            if vectorId is None or settings.HAS_WEAVAITEDB is False:
                result, conversessionID = model_predict_retry(user_id, tempid)
            else:
                result, conversessionID = rag_predict_retry(user_id, tempid)
            
            return Response({"result": result, "CCID": conversessionID}, status=status.HTTP_200_OK)
        else:
            return Response(
                {"redirect_url": "/login/"}, status=status.HTTP_403_FORBIDDEN
            )  # Return a redirect URL and 403 status
    except Exception as e:
        print(e)
        traceback.print_exc()
        return Response(
            {"result": settings.SYSTEM_PROMPTS["error_message"]},
            status=status.HTTP_200_OK,
        )


@api_view(["POST"])
def chat_undo(request):
    try:
        if request.session.get("is_authenticated"):
            user_id = request.session.get("user_id")
            tempid = request.data.get("CCID")
            result = message_undo(user_id, tempid)
            return Response({"result": result}, status=status.HTTP_200_OK)

        else:
            return Response(
                {"redirect_url": "/login/"}, status=status.HTTP_403_FORBIDDEN
            )  # Return a redirect URL and 403 status
    except Exception as e:
        print(e)
        traceback.print_exc()
        return Response(
            {"result": settings.SYSTEM_PROMPTS["error_message"]},
            status=status.HTTP_200_OK,
        )


def chat_view(request):
    if settings.HAS_WEAVAITEDB is True:
        return render(request, "chat.html")
    else:
        return render(request, "chat_no_upload.html")
@api_view(["POST"])
def get_all_chat_history(request):
    try:
        if request.session.get("is_authenticated"):
            user_id = request.session.get(
                "user_id"
            )  # Retrieve the user_id from session
            history_list = list_memory_sessions(user_id)
            return Response({"result": history_list}, status=status.HTTP_200_OK)

        else:
            return Response(
                {"redirect_url": "/login/"}, status=status.HTTP_403_FORBIDDEN
            )  # Return a redirect URL and 403 status
    except Exception as e:
        print(e)
        traceback.print_exc()
        return Response(
            {"result": settings.SYSTEM_PROMPTS["error_message"]},
            status=status.HTTP_200_OK,
        )
        
@api_view(["POST"])
def load_chat_history(request):
    try:
        if request.session.get("is_authenticated"):
            user_id = request.session.get(
                "user_id"
            )
            tempid = request.data.get("CCID")
            history_list = get_memory(user_id,tempid)
            vectorId=get_mapping(tempid)
            if history_list is None:
                return Response({"result": [],"vectorID":vectorId}, status=status.HTTP_200_OK)
            else:
                return Response({"result": history_list,"vectorID":vectorId}, status=status.HTTP_200_OK)

        else:
            return Response(
                {"redirect_url": "/login/"}, status=status.HTTP_403_FORBIDDEN
            )  # Return a redirect URL and 403 status
    except Exception as e:
        print(e)
        traceback.print_exc()
        return Response(
            {"result": settings.SYSTEM_PROMPTS["error_message"]},
            status=status.HTTP_200_OK,
        )


@api_view(["POST"])
def reset_chat_history(request):
    try:
        if request.session.get("is_authenticated"):
            user_id = request.session.get(
                "user_id"
            )
            tempid = request.data.get("CCID")
            del_Result=delete_memory(user_id,tempid)
            if del_Result:
                return Response(status=status.HTTP_200_OK)
            else:
                return Response(status=status.HTTP_304_NOT_MODIFIED)

        else:
            return Response(
                {"redirect_url": "/login/"}, status=status.HTTP_403_FORBIDDEN
            )  # Return a redirect URL and 403 status
    except Exception as e:
        print(e)
        traceback.print_exc()
        return Response(
            {"result": settings.SYSTEM_PROMPTS["error_message"]},
            status=status.HTTP_200_OK,
        )


def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        # Authenticate the user
        user = JsonFileBackend().authenticate(
            request, username=username, password=password
        )

        if user is not None:
            # Manually set a session key based on the username (or any unique field)
            request.session["user_id"] = username  # Store username in session

            # Also set the backend for the user object
            user.backend = "llm_app.auth_backend.JsonFileBackend"

            # Log the user in manually by setting session data
            request.session["is_authenticated"] = True

            # Redirect to the chatbot page or another success page
            return redirect("/chat/")  # Replace 'chatbot' with your success URL
        else:
            # Invalid login attempt
            return render(request, "login.html", {"error": "Invalid credentials"})

    return render(request, "login.html")

@api_view(['POST'])
def upload_pdf(request):
    if not request.session.get("is_authenticated"):
        return JsonResponse({"error": "Authentication required."}, status=403)
    # Generate timestamp if not provided
    timestamp=request.data.get("CCID")
    if timestamp is None or timestamp == '':
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        user_id = request.session.get("user_id")
        if 'pdf_files' not in request.FILES:
            return JsonResponse({"error": "No files uploaded."}, status=400)
        
        upload_dir = os.path.join('uploads', str(timestamp))
        os.makedirs(upload_dir, exist_ok=True)
        
        saved_files = []
        for file in request.FILES.getlist('pdf_files'):
            file_path = os.path.join(upload_dir, file.name)
            path = default_storage.save(file_path, ContentFile(file.read()))
            saved_files.append(path)
        newVector(timestamp,upload_dir)
        return JsonResponse({"message": "PDF(s) uploaded successfully!", "files": saved_files,"CCID":timestamp}, status=200)

    except Exception as e:
        traceback.print_exc()
        print(f"Error uploading PDFs: {e}")
        return JsonResponse({"error": "Failed to upload PDF(s)"}, status=500)

