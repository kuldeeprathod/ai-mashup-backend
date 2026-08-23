from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import uuid

app = FastAPI(title="AI Mashup Maker")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "AI Mashup Maker Backend"
    }


@app.post("/upload")
async def upload_songs(files: list[UploadFile] = File(...)):

    if len(files) < 2:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Please upload at least 2 songs."
            }
        )

    uploaded = []

    for file in files:

        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        path = os.path.join(UPLOAD_DIR, filename)

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded.append({
            "original_name": file.filename,
            "file": path
        })

    return {
        "status": "uploaded",
        "songs": uploaded
    }
