from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydub import AudioSegment
from groq import Groq
import os
import uuid
import json

app = FastAPI(title="AI Mashup Maker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "/tmp/mashup_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_client():
    key = os.environ.get("GROQ_API_KEY")

    if not key:
        raise Exception("GROQ_API_KEY is not configured.")

    return Groq(api_key=key)


def choose_best_line(segments):
    """
    Automatically chooses a meaningful spoken/sung segment.

    Preferred duration: 5-15 seconds.
    If a natural complete segment is longer than 15 seconds,
    it can be kept longer rather than cutting it artificially.
    """

    candidates = []

    for segment in segments:

        start = float(segment["start"])
        end = float(segment["end"])
        duration = end - start

        if duration < 5:
            continue

        # Prefer 5-15 second natural segments.
        if 5 <= duration <= 15:
            score = 100 + duration
        else:
            # Longer complete segment is still allowed.
            score = 50 - abs(duration - 12)

        candidates.append(
            {
                "start": start,
                "end": end,
                "duration": duration,
                "text": segment.get("text", "").strip(),
                "score": score
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[0]


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "AI Mashup Maker",
        "version": "4.0"
    }


@app.post("/analyze")
async def analyze_song(
    file: UploadFile = File(...)
):

    job_id = str(uuid.uuid4())
    extension = os.path.splitext(
        file.filename or ".mp3"
    )[1]

    input_path = f"/tmp/{job_id}{extension}"

    try:

        data = await file.read()

        # Groq currently accepts files up to 25 MB.
        if len(data) > 25 * 1024 * 1024:
            return {
                "status": "error",
                "message":
                "Audio file is larger than 25 MB."
            }

        with open(input_path, "wb") as f:
            f.write(data)

        client = get_client()

        with open(input_path, "rb") as audio_file:

            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        segments = []

        for segment in transcription.segments:

            if isinstance(segment, dict):
                start = segment["start"]
                end = segment["end"]
                text = segment.get("text", "")
            else:
                start = segment.start
                end = segment.end
                text = segment.text

            segments.append({
                "start": float(start),
                "end": float(end),
                "text": text.strip()
            })

        best = choose_best_line(segments)

        return {
            "status": "success",
            "filename": file.filename,
            "language": getattr(
                transcription,
                "language",
                None
            ),
            "segments": segments,
            "best_line": best
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

    finally:

        if os.path.exists(input_path):
            os.remove(input_path)


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

    job_id = str(uuid.uuid4())
    clips = []
    selected = []

    try:

        client = get_client()

        for index, upload in enumerate(files):

            extension = os.path.splitext(
                upload.filename or ".mp3"
            )[1]

            input_path = (
                f"/tmp/{job_id}_{index}{extension}"
            )

            data = await upload.read()

            if len(data) > 25 * 1024 * 1024:
                return {
                    "status": "error",
                    "message":
                    f"{upload.filename} is larger than 25 MB."
                }

            with open(input_path, "wb") as f:
                f.write(data)

            # -----------------------------
            # TRANSCRIPTION
            # -----------------------------

            with open(input_path, "rb") as audio_file:

                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

            segments = []

            for segment in transcription.segments:

                if isinstance(segment, dict):
                    start = segment["start"]
                    end = segment["end"]
                    text = segment.get("text", "")
                else:
                    start = segment.start
                    end = segment.end
                    text = segment.text

                segments.append({
                    "start": float(start),
                    "end": float(end),
                    "text": text.strip()
                })

            best = choose_best_line(segments)

            # -----------------------------
            # LOAD ORIGINAL AUDIO
            # -----------------------------

            audio = AudioSegment.from_file(
                input_path
            )

            if best:

                start_ms = int(
                    best["start"] * 1000
                )

                end_ms = int(
                    best["end"] * 1000
                )

                clip = audio[
                    start_ms:end_ms
                ]

                selected.append({
                    "song": upload.filename,
                    "text": best["text"],
                    "start": best["start"],
                    "end": best["end"],
                    "duration":
                    round(best["duration"], 2)
                })

            else:

                # Fallback:
                # middle 10 seconds
                duration = len(audio)

                clip_length = min(
                    10000,
                    duration
                )

                start_ms = max(
                    0,
                    (duration - clip_length) // 2
                )

                end_ms = start_ms + clip_length

                clip = audio[
                    start_ms:end_ms
                ]

                selected.append({
                    "song": upload.filename,
                    "text": "",
                    "start":
                    round(start_ms / 1000, 2),
                    "end":
                    round(end_ms / 1000, 2),
                    "duration":
                    round(len(clip) / 1000, 2)
                })

            clips.append(clip)

            if os.path.exists(input_path):
                os.remove(input_path)


        # -----------------------------
        # MERGE ORIGINAL AUDIO
        # -----------------------------

        mashup = AudioSegment.empty()

        for clip in clips:
            mashup += clip


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
            "message":
            "AI Mashup created successfully.",
            "selected_lines": selected,
            "download_url":
            f"/download/{job_id}"
        }


    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/download/{job_id}")
def download_mashup(
    job_id: str
):

    path = os.path.join(
        OUTPUT_DIR,
        f"{job_id}.mp3"
    )

    if not os.path.exists(path):

        return {
            "status": "error",
            "message":
            "Mashup file not found."
        }

    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename="AI-Mashup.mp3"
    )
