from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "AI Mashup Maker"
    }


@app.post("/create-mashup")
async def create_mashup(
    files: list[UploadFile] = File(...)
):

    return {
        "status": "success",
        "message": "Upload endpoint is working",
        "number_of_files": len(files),
        "files": [
            file.filename
            for file in files
        ]
    }
