from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydub import AudioSegment
import os
import uuid
import shutil

app = FastAPI(title="AI Mashup Maker")

BASE_DIR = "mashup_data"
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "AI Mashup Maker Backend"
    }


@app.post("/create-mashup")
async def create_mashup(files: list[UploadFile] = File(...)):

    if len(files) < 2:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Please upload at least 2 songs."
            }
        )

    job_id = str(uuid.uuid4())

    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    clips = []

    try:

        # Save uploaded songs
        for index, file in enumerate(files):

            filename = file.filename or f"song_{index}.mp3"

            safe_name = f"{index}_{filename}"

            input_path = os.path.join(
                job_dir,
                safe_name
            )

            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            # Load original audio
            audio = AudioSegment.from_file(
                input_path
            )

            # ------------------------------------------------
            # TEMPORARY MVP SELECTION
            # Later AI will select the real best line.
            # ------------------------------------------------

            clip_length = min(
                15 * 1000,
                len(audio)
            )

            start = max(
                0,
                len(audio) // 2 - clip_length // 2
            )

            end = start + clip_length

            clip = audio[start:end]

            clips.append(clip)


        # Merge original clips
        mashup = AudioSegment.empty()

        for clip in clips:

            mashup += clip


        # Export final MP3
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{job_id}.mp3"
        )

        mashup.export(
            output_path,
            format="mp3",
            bitrate="192k"
        )


        return {
            "status": "success",
            "job_id": job_id,
            "download_url":
                f"/download/{job_id}"
        }


    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.get("/download/{job_id}")
def download_mashup(job_id: str):

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}.mp3"
    )

    if not os.path.exists(output_path):

        return JSONResponse(
            status_code=404,
            content={
                "error": "Mashup not found."
            }
        )

    return FileResponse(
        output_path,
        media_type="audio/mpeg",
        filename="AI-Mashup.mp3"
    )
