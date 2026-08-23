from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Mashup Maker")

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
        "service": "AI Mashup Maker",
        "version": "2.0"
    }


@app.post("/analyze")
async def analyze_song(file: UploadFile = File(...)):

    return {
        "status": "ready",
        "message": "Audio analysis endpoint is ready",
        "filename": file.filename
    }


@app.post("/create-mashup")
async def create_mashup(
    files: list[UploadFile] = File(...)
):

    if len(files) < 2:
        return {
            "status": "error",
            "message": "Upload at least 2 songs."
        }

    return {
        "status": "ready",
        "message": "Mashup request received",
        "songs": [
            file.filename for file in files
        ]
    }
