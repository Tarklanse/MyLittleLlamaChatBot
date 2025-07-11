from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from .service import (
    model_predict_retry,
    model_predict,
    message_undo,
    rag_predict,
    rag_predict_retry,
    rag_predict_openai,
    rag_predict_retry_openai,
)
from .memory_handler import (
    list_memory_sessions,
    delete_memory,
    get_memory,
    edit_persenal,
)
from .user_handler import (
    query_user,
    get_all_user,
    add_user,
    del_user,
    edit_user,
    encode_string,
)
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
from .weaviateVectorStoreHandler import newVector,get_vector_list
from .vectorMapper import get_mapping,get_All_mapping
from django.http import FileResponse, Http404


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
            vectorId = get_mapping(tempid)
            if vectorId is None or settings.HAS_WEAVAITEDB is False:
                result, conversessionID = model_predict(prompt, user_id, tempid)
            elif settings.MODEL_TYPE == "openai" and vectorId is not None:
                result, conversessionID = rag_predict_openai(prompt, user_id, tempid)
            elif settings.MODEL_TYPE == "api" and vectorId is not None:
                result, conversessionID = rag_predict_openai(prompt, user_id, tempid)
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
            vectorId = get_mapping(tempid)
            if vectorId is None or settings.HAS_WEAVAITEDB is False:
                result, conversessionID = model_predict_retry(user_id, tempid)
            elif settings.MODEL_TYPE == "openai" and vectorId is not None:
                result, conversessionID = rag_predict_retry_openai(user_id, tempid)
            else:
                result, conversessionID = rag_predict_retry(user_id, tempid)

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
def edit_Personal(request):
    try:
        if request.session.get("is_authenticated"):
            user_id = request.session.get("user_id")
            tempid = request.data.get("CCID")
            newPersonal = request.data.get("Personal")
            if newPersonal is None or newPersonal == "":
                newPersonal = settings.SYSTEM_PROMPTS["Default_Personal"]
            tempid = edit_persenal(user_id, tempid, newPersonal)
            return Response({"CCID": tempid}, status=status.HTTP_200_OK)
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
            user_id = request.session.get("user_id")
            tempid = request.data.get("CCID")
            history_list = get_memory(user_id, tempid)
            vectorId = get_mapping(tempid)
            if history_list is None:
                return Response(
                    {"result": [], "vectorID": vectorId}, status=status.HTTP_200_OK
                )
            else:
                return Response(
                    {"result": history_list, "vectorID": vectorId},
                    status=status.HTTP_200_OK,
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
def reset_chat_history(request):
    try:
        if request.session.get("is_authenticated"):
            user_id = request.session.get("user_id")
            tempid = request.data.get("CCID")
            del_Result = delete_memory(user_id, tempid)
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
            request, username=username, password=encode_string(password)
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


@api_view(["POST"])
def upload_pdf(request):
    if not request.session.get("is_authenticated"):
        return JsonResponse({"error": "Authentication required."}, status=403)
    # Generate timestamp if not provided
    timestamp = request.data.get("CCID")
    if timestamp is None or timestamp == "":
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        user_id = request.session.get("user_id")
        if "pdf_files" not in request.FILES:
            return JsonResponse({"error": "No files uploaded."}, status=400)

        upload_dir = os.path.join("uploads", str(timestamp))
        os.makedirs(upload_dir, exist_ok=True)

        saved_files = []
        for file in request.FILES.getlist("pdf_files"):
            file_path = os.path.join(upload_dir, file.name)
            path = default_storage.save(file_path, ContentFile(file.read()))
            saved_files.append(path)
        newVector(timestamp, upload_dir)
        return JsonResponse(
            {
                "message": "PDF(s) uploaded successfully!",
                "files": saved_files,
                "CCID": timestamp,
            },
            status=200,
        )

    except Exception as e:
        traceback.print_exc()
        print(f"Error uploading PDFs: {e}")
        return JsonResponse({"error": "Failed to upload PDF(s)"}, status=500)


@api_view(["GET"])
def download_AI_Gen_file(request, filename):
    """
    Serve a file for download.

    Parameters:
    - request: The HTTP request object.
    - filename: The name of the file to be downloaded.

    Returns:
    - FileResponse: A response that streams the file to the user.
    """

    file_path = os.path.join(settings.AI_TEXT_PATH, filename)
    if not os.path.exists(file_path):
        raise Http404("File does not exist")

    response = FileResponse(open(file_path, "rb"), as_attachment=True)
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


# manage page
def login_view_manage(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        # Authenticate the user
        user = JsonFileBackend().authenticate(
            request, username=username, password=encode_string(password)
        )

        if user is not None:
            if user.role != "admin":
                return render(
                    request,
                    "login.html",
                    {"error": "You don't have role to login manage page"},
                )
            # Manually set a session key based on the username (or any unique field)
            request.session["user_id"] = username

            user.backend = "llm_app.auth_backend.JsonFileBackend"

            request.session["is_authenticated"] = True

            return redirect("/manage/users")
        else:
            # Invalid login attempt
            return render(
                request, "login_manage.html", {"error": "Invalid credentials"}
            )

    return render(request, "login_manage.html")

@api_view(["GET", "POST"])
def manage_vector_view(request):
    if not request.session.get("is_authenticated"):
        return JsonResponse({"error": "Authentication required."}, status=403)
    if request.method == "GET":
        # Fetch all users to display
        #users = get_all_user()
        vector=get_vector_list()
        vectors=get_All_mapping()
        return render(request, "manage_vector.html", {"users": vector})

    elif request.method == "POST":
        action = request.data["action"]
        if action == "delete":
            username = request.data["username"]
            result = del_user(username)
            if result:
                response = {"status": "success", "message": "user deleted"}
            else:
                response = {"status": "success", "message": "user not deleted"}
        else:
            response = {"status": "error", "message": "Invalid action"}

        return JsonResponse(response)
@api_view(["GET", "POST"])
def manage_users_view(request):
    if not request.session.get("is_authenticated"):
        return JsonResponse({"error": "Authentication required."}, status=403)
    if request.method == "GET":
        # Fetch all users to display
        users = get_all_user()
        #get_vector_list()
        return render(request, "manage_users.html", {"users": users})

    elif request.method == "POST":
        action = request.data["action"]
        if action == "add":
            username = request.data["username"]
            password = request.data["password"]
            role = request.data["role"]
            result = add_user(username, password, role)
            if result:
                response = {"status": "success", "message": "user added"}
            else:
                response = {"status": "success", "message": "user not added"}
        elif action == "delete":
            username = request.data["username"]
            result = del_user(username)
            if result:
                response = {"status": "success", "message": "user deleted"}
            else:
                response = {"status": "success", "message": "user not deleted"}
        elif action == "edit":
            # Editing is a combination of delete and add
            old_username = request.data["old_username"]
            new_username = request.data["new_username"]
            password = request.data["new_password"]
            role = request.data["role"]
            result = edit_user(old_username, new_username, password, role)
            if result:
                response = {"status": "success", "message": "user edited"}
            else:
                response = {"status": "success", "message": "user not edited"}
        else:
            response = {"status": "error", "message": "Invalid action"}

        return JsonResponse(response)
