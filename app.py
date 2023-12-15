from long_trainer import LongTrainer
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
import json
import os
import shutil
import uuid
import datetime
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md", ".markdown", ".html", ".csv", ".docx", ".jpg", ".jpeg", ".png"}

def is_allowed_file(filename):
    return any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)

os.environ["OPENAI_API_KEY"] = "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"
trainer = LongTrainer()

app = FastAPI()

class DeleteModelRequest(BaseModel):
    bot_id: str

class QueryData(BaseModel):
    query: str
    bot_id: str
    chat_id: str

def get_model_id(paths, links, bot_id):
    try:
        for path in paths:
            trainer.add_document_from_path(path, bot_id)
        trainer.add_document_from_link(links, bot_id)
        trainer.create_bot(bot_id)
        chat_id = trainer.new_chat(bot_id)
        vision_chat_id = trainer.new_vision_chat(bot_id)
        return chat_id, vision_chat_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def response(query, bot_id, chat_id):
    try:
        response = trainer._get_response(query, chat_id, bot_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def vision_response(query, image_paths, vision_chat_id, bot_id):

    print(query)
    print(image_paths)
    print(vision_chat_id)
    print(bot_id)

    result = trainer._get_vision_response(query, image_paths, str(vision_chat_id), str(bot_id))
    return result

def update_model(paths, links, bot_id):
    print(paths)
    print(links)
    print(bot_id)
    try:
        trainer.update_chatbot(paths, links, None, bot_id)
        chat_id = trainer.new_chat(bot_id)
        vision_chat_id = trainer.new_vision_chat(bot_id)
        return chat_id, vision_chat_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def perform_model_deletion(bot_id):
    try:
        trainer.delet_chatbot(bot_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_uploaded_files(files: List[UploadFile], bot_id: str):
    try:
        bot_folder = f"./uploads/{bot_id}/files"
        os.makedirs(bot_folder, exist_ok=True)
        paths = []
        for file in files:
            if not is_allowed_file(file.filename):
                continue
            path = os.path.join(bot_folder, file.filename)
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            paths.append(path)
        return paths
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_uploaded_images(files: List[UploadFile], bot_id: str):
    try:
        bot_folder = f"./uploads/{bot_id}/images"
        os.makedirs(bot_folder, exist_ok=True)
        image_paths = []
        for file in files:
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(bot_folder, file.filename)
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_paths.append(path)
        return image_paths
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





def save_uploaded_json_file(json_file: UploadFile):
    common_folder = "./uploads/common/json_file"
    os.makedirs(common_folder, exist_ok=True)

    unique_number = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + "-" + str(uuid.uuid4())
    filename = f"{unique_number}.json"
    path = os.path.join(common_folder, filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(json_file.file, buffer)
    return path



@app.post("/deploy_model/")
async def deploy_model(files: List[UploadFile] = File(...), links: List[str] = Form(...)):
    bot_id = trainer.initialize_bot_id()
    paths = save_uploaded_files([file for file in files if is_allowed_file(file.filename)], bot_id)
    chat_id, vision_chat_id = get_model_id(paths, links, bot_id)
    return {"bot_id": bot_id, "chat_id": chat_id, "vision_chat_id": vision_chat_id}

@app.post("/response/")
async def get_response(query_data: QueryData):
    return {"response": response(query_data.query, query_data.bot_id, query_data.chat_id)}

@app.post("/vision_response/")
async def get_vision_response(json_file: UploadFile = File(...), files: List[UploadFile] = File(None)):
    # Read and parse the JSON file
    json_file_path = save_uploaded_json_file(json_file)
    with open(json_file_path, "r") as file:
        loaded_data = json.load(file)

    bot_id = loaded_data.get('bot_id')
    vision_chat_id = loaded_data.get('vision_chat_id')
    query = loaded_data.get('query')

    print('files : ', files)

    if bot_id and vision_chat_id:
        if files:
            image_paths = save_uploaded_images(files, bot_id)
            print(image_paths)
            return {"vision_response": vision_response(query, image_paths, str(vision_chat_id), str(bot_id))}
        else:
            return {"vision_response": vision_response(query, [], str(vision_chat_id), str(bot_id))}




@app.post("/update_model/")
async def update_bot_model(json_file: UploadFile = File(...), files: List[UploadFile] = File(None),
                           links: List[str] = Form(None)):



    json_file_path = save_uploaded_json_file(json_file)

    with open(json_file_path, "r") as file:
        loaded_data = json.load(file)
    # If you need to immediately load and use the saved JSON file

    bot_id = loaded_data.get('bot_id')
    if not bot_id:
        raise HTTPException(status_code=400, detail="Bot ID is required")

    paths = save_uploaded_files([file for file in files if is_allowed_file(file.filename)], bot_id)


    # Your model update logic here
    chat_id, vision_chat_id = update_model(paths, links, bot_id)
    return {"new_chat_id": chat_id, "new_vision_chat_id": vision_chat_id}


@app.delete("/delete_model/")
async def remove_model(request_data: DeleteModelRequest):
    bot_id = request_data.bot_id
    try:
        perform_model_deletion(bot_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))