from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydub import AudioSegment
from groq import Groq
import os
import uuid
import json
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
    return re.sub(r"\s+", " ", text.strip())


def transcribe_audio(client, file_path):

    with open(file_path, "rb") as audio_file:

        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            language="hi",
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

        text = clean_text(text)

        if text:
            segments.append({
                "start": float(start),
                "end": float(end),
                "text": text
            })

    return segments


def build_natural_lines(segments):

    lines = []

    current = None

    for segment in segments:

        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        if current is None:

            current = {
                "start": start,
                "end": end,
                "text": text
            }

            continue

        gap = start - current["end"]

        current_duration = (
            current["end"] -
            current["start"]
        )

        # Join closely connected speech.
        if gap <= 0.8 and current_duration < 25:

            current["end"] = end

            current["text"] = (
                current["text"] +
                " " +
                text
            ).strip()

        else:

            lines.append(current)

            current = {
                "start": start,
                "end": end,
                "text": text
            }

    if current:
        lines.append(current)

    return lines


def local_score(line):

    duration = (
        line["end"] -
        line["start"]
    )

    words = line["text"].split()

    score = 0

    # Preferred duration.
    if 5 <= duration <= 15:
        score += 40

    elif 15 < duration <= 22:
        score += 30

    elif duration > 22:
        score += 10

    else:
        score += 5

    # Useful amount of text.
    if 6 <= len(words) <= 25:
        score += 30

    elif len(words) >= 4:
        score += 15

    # Avoid obvious filler.
    filler = [
        "ओह",
        "आह",
        "हम्म",
        "yeah",
        "yo",
        "oh",
        "aah",
        "hmm",
        "la",
        "na"
    ]

    filler_count = 0

    for word in words:
        if word.lower() in filler:
            filler_count += 1

    score -= filler_count * 10

    return score


def rank_lines_with_ai(client, lines):

    if not lines:
        return None

    candidates = []

    for index, line in enumerate(lines):

        duration = (
            line["end"] -
            line["start"]
        )

        score = local_score(line)

        if duration >= 3 and score >= 10:

            candidates.append({
                "index": index,
                "start": line["start"],
                "end": line["end"],
                "duration": round(duration, 2),
                "text": line["text"],
                "local_score": score
            })

    if not candidates:
        return None

    # Limit prompt size.
    candidates = sorted(
        candidates,
        key=lambda x: x["local_score"],
        reverse=True
    )[:30]

    candidate_text = ""

    for item in candidates:

        candidate_text += (
            f'\nID {item["index"]} | '
            f'{item["duration"]} sec | '
            f'{item["text"]}'
        )

    prompt = f"""
You are an expert Hindi music mashup editor.

Choose ONE candidate line that will work best inside a Hindi song mashup.

Priorities:
1. Meaningful and memorable lyric.
2. Strong emotional or catchy hook.
3. Sounds natural when isolated.
4. Prefer approximately 5-15 seconds.
5. If a complete natural line is longer than 15 seconds,
   DO NOT reject it only because it is longer.
6. Avoid filler vocals such as oh, yeah, aah, na, etc.
7. Do not choose an incomplete phrase if another complete line exists.

Return ONLY valid JSON:

{{"id": NUMBER}}

Candidates:
{candidate_text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    # Extract JSON safely.
    match = re.search(
        r'\{.*?\}',
        content,
        re.DOTALL
    )

    if not match:
        return candidates[0]

    try:

        result = json.loads(
            match.group(0)
        )

        selected_id = int(
            result["id"]
        )

        for item in candidates:

            if item["index"] == selected_id:
                return item

    except Exception:
        pass

    return candidates[0]


def analyze_file(client, file_path):

    segments = transcribe_audio(
        client,
        file_path
    )

    natural_lines = build_natural_lines(
        segments
    )

    best = rank_lines_with_ai(
        client,
        natural_lines
    )

    return {
        "segments": segments,
        "lines": natural_lines,
        "best_line": best
    }


@app.get("/")
def home():

    return {
        "status": "online",
        "service": "AI Mashup Maker",
        "version": "7.0"
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
                "status": "error",
                "message":
                "Audio file is larger than 25 MB."
            }

        with open(
            input_path,
            "wb"
        ) as f:

            f.write(data)

        client = get_client()

        result = analyze_file(
            client,
            input_path
        )

        return {
            "status": "success",
            "filename": file.filename,
            "segments": result["segments"],
            "lines": result["lines"],
            "best_line": result["best_line"]
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
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
            "status": "error",
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
                    "status": "error",
                    "message":
                    f"{upload.filename} "
                    "is larger than 25 MB."
                }

            with open(
                input_path,
                "wb"
            ) as f:

                f.write(data)

            result = analyze_file(
                client,
                input_path
            )

            best = result["best_line"]

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
                        best["duration"]
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

            clips.append(clip)

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
        media_type="audio/mpeg",
        filename="AI-Mashup.mp3"
    )
