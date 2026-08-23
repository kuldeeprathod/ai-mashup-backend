from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import uuid

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
        "version": "3.0"
    }


@app.post("/analyze")
async def analyze_song(
    file: UploadFile = File(...)
):

    temp_name = (
        "/tmp/"
        + str(uuid.uuid4())
        + "_"
        + (file.filename or "song.mp3")
    )

    try:

        data = await file.read()

        with open(temp_name, "wb") as f:
            f.write(data)

        audio = AudioSegment.from_file(
            temp_name
        )

        mono = audio.set_channels(1)

        if mono.dBFS == float("-inf"):
            threshold = -40
        else:
            threshold = max(
                -45,
                mono.dBFS - 16
            )

        ranges = detect_nonsilent(
            mono,
            min_silence_len=450,
            silence_thresh=threshold
        )

        sections = []

        for start, end in ranges:

            duration = end - start

            if duration < 5000:
                continue

            sections.append({
                "start_ms": start,
                "end_ms": end,
                "duration_seconds":
                    round(duration / 1000, 2)
            })

        return {
            "status": "success",
            "filename": file.filename,
            "duration_seconds":
                round(len(audio) / 1000, 2),
            "sections": sections
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

    finally:

        if os.path.exists(temp_name):
            os.remove(temp_name)


@app.post("/create-mashup")
async def create_mashup(
    files: list[UploadFile] = File(...)
):

    if len(files) < 2:

        return {
            "status": "error",
            "message":
                "Please upload at least 2 songs."
        }

    return {
        "status": "success",
        "message":
            "Mashup request received.",
        "songs": [
            file.filename
            for file in files
        ]
    }
