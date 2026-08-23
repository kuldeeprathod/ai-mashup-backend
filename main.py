from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

import os
import uuid
import shutil


app = FastAPI(title="AI Mashup Maker")


# --------------------------------------------------
# CORS - Blogger website ko backend access dene ke liye
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Folders
# --------------------------------------------------

BASE_DIR = "mashup_data"

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "status": "online",
        "service": "AI Mashup Maker",
        "version": "1.0"
    }


# --------------------------------------------------
# Find Best Natural Audio Section
# --------------------------------------------------

def find_best_section(audio):

    duration = len(audio)

    if duration <= 15000:

        return 0, duration


    # Audio ko mono analysis ke liye use karenge.
    # Original audio modify nahi hoga.
    analysis_audio = audio.set_channels(1)

    # Relative silence threshold
    if analysis_audio.dBFS == float("-inf"):

        threshold = -40

    else:

        threshold = max(
            -45,
            analysis_audio.dBFS - 16
        )


    # Non-silent portions detect
    ranges = detect_nonsilent(
        analysis_audio,
        min_silence_len=450,
        silence_thresh=threshold
    )


    if not ranges:

        # Agar detection fail ho,
        # middle section choose karenge.
        clip_length = min(
            15000,
            duration
        )

        start = max(
            0,
            (duration - clip_length) // 2
        )

        return start, start + clip_length


    candidates = []


    for start, end in ranges:

        length = end - start

        # Bahut chhote noise sections ignore
        if length < 5000:
            continue

        # Natural section ko thoda expand karna
        start = max(
            0,
            start - 250
        )

        end = min(
            duration,
            end + 250
        )

        length = end - start

        candidates.append(
            (start, end, length)
        )


    if not candidates:

        clip_length = min(
            15000,
            duration
        )

        start = max(
            0,
            (duration - clip_length) // 2
        )

        return start, start + clip_length


    # Best candidate:
    # meaningful long section ko preference
    candidates.sort(
        key=lambda x: x[2],
        reverse=True
    )


    start, end, length = candidates[0]


    # Normally 5-15 sec target.
    # Lekin natural section 15 sec se bada ho
    # to usko forcefully cut nahi karenge.
    if length <= 15000:

        return start, end


    # Agar section bahut bada hai,
    # 15 sec ke around natural point choose karna.
    target = 15000

    middle = (start + end) // 2

    new_start = max(
        start,
        middle - target // 2
    )

    new_end = min(
        end,
        new_start + target
    )


    return new_start, new_end


# --------------------------------------------------
# Create Mashup
# --------------------------------------------------

@app.post("/create-mashup")
async def create_mashup(
    files: list[UploadFile] = File(...)
):

    if len(files) < 2:

        return JSONResponse(
            status_code=400,
            content={
                "error":
                "Please upload at least 2 songs."
            }
        )


    job_id = str(uuid.uuid4())


    job_dir = os.path.join(
        UPLOAD_DIR,
        job_id
    )

    os.makedirs(
        job_dir,
        exist_ok=True
    )


    clips = []

    selected_sections = []


    try:

        # ------------------------------------------
        # Process every uploaded song
        # ------------------------------------------

        for index, file in enumerate(files):

            original_name = (
                file.filename
                or f"song_{index}.mp3"
            )


            extension = os.path.splitext(
                original_name
            )[1].lower()


            if extension not in [
                ".mp3",
                ".wav",
                ".m4a",
                ".aac",
                ".ogg",
                ".flac"
            ]:

                continue


            input_path = os.path.join(
                job_dir,
                f"{index}{extension}"
            )


            # Save original uploaded file
            with open(
                input_path,
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )


            # Load audio
            audio = AudioSegment.from_file(
                input_path
            )


            # --------------------------------------
            # Find natural section
            # --------------------------------------

            start, end = find_best_section(
                audio
            )


            # --------------------------------------
            # Cut ORIGINAL audio
            # --------------------------------------

            clip = audio[
                start:end
            ]


            clips.append(
                clip
            )


            selected_sections.append({

                "song":
                original_name,

                "start_ms":
                start,

                "end_ms":
                end,

                "duration_seconds":
                round(
                    (end - start) / 1000,
                    2
                )

            })


        if len(clips) < 2:

            return JSONResponse(
                status_code=400,
                content={
                    "error":
                    "Could not process enough audio files."
                }
            )


        # ------------------------------------------
        # MERGE
        # ------------------------------------------

        mashup = AudioSegment.empty()


        for clip in clips:

            # No pitch change
            # No speed change
            # No effects
            # No remix

            mashup += clip


        # ------------------------------------------
        # Export
        # ------------------------------------------

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

            "status":
            "success",

            "job_id":
            job_id,

            "message":
            "Mashup created successfully.",

            "sections":
            selected_sections,

            "download_url":
            f"/download/{job_id}"

        }


    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "status":
                "error",

                "message":
                str(e)
            }
        )


# --------------------------------------------------
# Download
# --------------------------------------------------

@app.get(
    "/download/{job_id}"
)
def download_mashup(
    job_id: str
):

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}.mp3"
    )


    if not os.path.exists(
        output_path
    ):

        return JSONResponse(
            status_code=404,
            content={
                "error":
                "Mashup not found."
            }
        )


    return FileResponse(

        output_path,

        media_type=
        "audio/mpeg",

        filename=
        "AI-Mashup.mp3"

    )
