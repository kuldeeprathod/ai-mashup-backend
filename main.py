from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydub import AudioSegment
from groq import Groq
import os
import uuid
import re

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


def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def score_line(text, duration):

    text = clean_text(text)
    words = text.split()

    if duration < 3 or len(words) < 3:
        return -1000

    score = 0

    if 5 <= duration <= 15:
        score += 100
    elif 15 < duration <= 25:
        score += 70
    else:
        score += 20

    if 5 <= len(words) <= 25:
        score += 40
    elif len(words) > 25:
        score += 15

    filler_words = [
        "yeah", "yo", "oh", "ooh",
        "aah", "hmm", "la", "na", "hey"
    ]

    for word in words:
        if word.lower() in filler_words:
            score -= 15

    if len(words) >= 6:
        score += 20

    if len(words) >= 10:
        score += 10

    return score


def choose_best_line(segments):

    candidates = []

    for segment in segments:

        start = float(segment["start"])
        end = float(segment["end"])
        text = segment.get("text", "").strip()

        duration = end - start

        score = score_line(
            text,
            duration
        )

        if score >= 0:

            candidates.append({
                "start": start,
                "end": end,
                "duration": duration,
                "text": text,
                "score": score
            })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[0]


def transcribe_audio(
    client,
    file_path
):

    with open(
        file_path,
        "rb"
    ) as audio_file:

        transcription = client.audio.transcriptions.create(

            file=audio_file,

            model="whisper-large-v3-turbo",

            language="hi",

            response_format="verbose_json",

            timestamp_granularities=[
                "segment"
            ]
        )

    segments = []

    for segment in transcription.segments:

        if isinstance(segment, dict):

            start = segment["start"]
            end = segment["end"]
            text = segment.get(
                "text",
                ""
            )

        else:

            start = segment.start
            end = segment.end
            text = segment.text

        segments.append({

            "start":
                float(start),

            "end":
                float(end),

            "text":
                clean_text(text)

        })

    return segments


@app.get("/")
def home():

    return {

        "status":
            "online",

        "service":
            "AI Mashup Maker",

        "version":
            "6.0 Hindi"

    }


@app.post("/analyze")
async def analyze_song(
    file: UploadFile = File(...)
):

    job_id = str(uuid.uuid4())

    extension = os.path.splitext(
        file.filename or ".mp3"
    )[1]

    input_path = (
        f"/tmp/{job_id}{extension}"
    )

    try:

        data = await file.read()

        if len(data) > 25 * 1024 * 1024:

            return {

                "status":
                    "error",

                "message":
                    "Audio file is larger than 25 MB."

            }

        with open(
            input_path,
            "wb"
        ) as f:

            f.write(data)

        client = get_client()

        segments = transcribe_audio(
            client,
            input_path
        )

        best = choose_best_line(
            segments
        )

        return {

            "status":
                "success",

            "filename":
                file.filename,

            "segments":
                segments,

            "best_line":
                best

        }

    except Exception as e:

        return {

            "status":
                "error",

            "message":
                str(e)

        }

    finally:

        if os.path.exists(
            input_path
        ):

            os.remove(
                input_path
            )


@app.post("/create-mashup")
async def create_mashup(
    files: list[UploadFile] = File(...)
):

    if len(files) < 2:

        return {

            "status":
                "error",

            "message":
                "Please upload at least 2 songs."

        }

    job_id = str(uuid.uuid4())

    clips = []

    selected_lines = []

    try:

        client = get_client()

        for index, upload in enumerate(files):

            extension = os.path.splitext(
                upload.filename or ".mp3"
            )[1]

            input_path = (
                f"/tmp/"
                f"{job_id}_{index}"
                f"{extension}"
            )

            data = await upload.read()

            if len(data) > 25 * 1024 * 1024:

                return {

                    "status":
                        "error",

                    "message":
                        f"{upload.filename} "
                        "is larger than 25 MB."

                }

            with open(
                input_path,
                "wb"
            ) as f:

                f.write(data)

            segments = transcribe_audio(
                client,
                input_path
            )

            best = choose_best_line(
                segments
            )

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

                selected_lines.append({

                    "song":
                        upload.filename,

                    "text":
                        best["text"],

                    "start":
                        best["start"],

                    "end":
                        best["end"],

                    "duration":
                        round(
                            best["duration"],
                            2
                        )

                })

            else:

                clip_length = min(
                    10000,
                    len(audio)
                )

                clip = audio[
                    0:clip_length
                ]

                selected_lines.append({

                    "song":
                        upload.filename,

                    "text":
                        "",

                    "start":
                        0,

                    "end":
                        round(
                            clip_length / 1000,
                            2
                        ),

                    "duration":
                        round(
                            clip_length / 1000,
                            2
                        )

                })

            clips.append(
                clip
            )

            if os.path.exists(
                input_path
            ):

                os.remove(
                    input_path
                )

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

            "status":
                "success",

            "job_id":
                job_id,

            "message":
                "AI Mashup created successfully.",

            "selected_lines":
                selected_lines,

            "download_url":
                f"/download/{job_id}"

        }

    except Exception as e:

        return {

            "status":
                "error",

            "message":
                str(e)

        }


@app.get(
    "/download/{job_id}"
)
def download_mashup(
    job_id: str
):

    path = os.path.join(

        OUTPUT_DIR,

        f"{job_id}.mp3"

    )

    if not os.path.exists(path):

        return {

            "status":
                "error",

            "message":
                "Mashup file not found."

        }

    return FileResponse(

        path,

        media_type=
            "audio/mpeg",

        filename=
            "AI-Mashup.mp3"

    )
