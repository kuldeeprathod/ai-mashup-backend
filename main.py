from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydub import AudioSegment
from groq import Groq
import os
import uuid
import json
import re
import subprocess
import shutil

app = FastAPI(title="AI Mashup Maker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "/tmp/mashup_outputs"
SEPARATION_DIR = "/tmp/demucs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEPARATION_DIR, exist_ok=True)

AI_MODEL = "openai/gpt-oss-120b"
WHISPER_MODEL = "whisper-large-v3-turbo"

CROSSFADE_MS = 1500
TARGET_DBFS = -16.0
MAX_CLIP_MS = 15000
MIN_CLIP_MS = 4000

VOCAL_GAIN = 1.0
INSTRUMENTAL_GAIN = 0.35


def get_client():
    key = os.environ.get("GROQ_API_KEY")

    if not key:
        raise Exception("GROQ_API_KEY is not configured.")

    return Groq(api_key=key)


def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())


def transcribe_audio(client, file_path):

    with open(file_path, "rb") as audio_file:

        result = client.audio.transcriptions.create(
            file=audio_file,
            model=WHISPER_MODEL,
            language="hi",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    segments = []

    for segment in result.segments:

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


def build_lines(segments):

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

        duration = (
            current["end"] -
            current["start"]
        )

        if gap <= 0.8 and duration < 25:

            current["end"] = end

            current["text"] = (
                current["text"]
                + " "
                + text
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


def score_line(line):

    duration = (
        line["end"] -
        line["start"]
    )

    words = line["text"].split()

    score = 0

    if 5 <= duration <= 15:
        score += 40

    elif 15 < duration <= 22:
        score += 30

    elif duration > 22:
        score += 10

    if 6 <= len(words) <= 25:
        score += 30

    elif len(words) >= 4:
        score += 15

    fillers = [
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

    for word in words:

        if word.lower() in fillers:
            score -= 10

    return score


def choose_best_line(client, lines):

    candidates = []

    for index, line in enumerate(lines):

        duration = (
            line["end"] -
            line["start"]
        )

        score = score_line(line)

        if duration >= 3:

            candidates.append({
                "index": index,
                "start": line["start"],
                "end": line["end"],
                "duration": round(duration, 2),
                "text": line["text"],
                "score": score
            })

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )[:30]

    text = ""

    for item in candidates:

        text += (
            f'\nID {item["index"]} | '
            f'{item["duration"]} sec | '
            f'{item["text"]}'
        )

    prompt = f"""
You are an expert Hindi music mashup editor.

Select ONE lyric section that will sound best
in a professional mashup.

Choose:
- catchy
- emotional
- memorable
- complete phrase
- approximately 5-15 seconds
- no filler
- no incomplete phrase

Return ONLY JSON.

Example:
{{"id": 12}}

Candidates:
{text}
"""

    response = client.chat.completions.create(
        model=AI_MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = (
        response.choices[0]
        .message
        .content
        .strip()
    )

    match = re.search(
        r'\{.*?\}',
        content,
        re.DOTALL
    )

    if match:

        try:

            result = json.loads(
                match.group(0)
            )

            selected = int(
                result["id"]
            )

            for item in candidates:

                if item["index"] == selected:
                    return item

        except Exception:
            pass

    return candidates[0]


def separate_audio(
    input_path,
    job_id
):

    output_dir = os.path.join(
        SEPARATION_DIR,
        job_id
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    command = [
        "python",
        "-m",
        "demucs",
        "--two-stems=vocals",
        "-d",
        "cpu",
        "-n",
        "htdemucs",
        "-o",
        output_dir,
        input_path
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800
    )

    if process.returncode != 0:

        raise Exception(
            "Demucs separation failed: "
            + process.stderr[-3000:]
        )

    song_name = os.path.splitext(
        os.path.basename(input_path)
    )[0]

    stem_dir = os.path.join(
        output_dir,
        "htdemucs",
        song_name
    )

    vocals = os.path.join(
        stem_dir,
        "vocals.wav"
    )

    no_vocals = os.path.join(
        stem_dir,
        "no_vocals.wav"
    )

    if not os.path.exists(vocals):

        raise Exception(
            "Demucs vocals.wav was not created."
        )

    if not os.path.exists(no_vocals):

        raise Exception(
            "Demucs no_vocals.wav was not created."
        )

    return vocals, no_vocals


def normalize(audio):

    if len(audio) == 0:
        return audio

    if audio.dBFS == float("-inf"):
        return audio

    gain = TARGET_DBFS - audio.dBFS

    gain = max(
        -8,
        min(8, gain)
    )

    return audio.apply_gain(gain)


def make_stem_clip(
    vocals_path,
    instrumental_path,
    best
):

    vocals = AudioSegment.from_file(
        vocals_path
    )

    instrumental = AudioSegment.from_file(
        instrumental_path
    )

    if best:

        start = int(
            best["start"] * 1000
        )

        end = int(
            best["end"] * 1000
        )

        vocal_clip = vocals[
            start:end
        ]

        instrumental_clip = instrumental[
            start:end
        ]

    else:

        vocal_clip = vocals[
            :MAX_CLIP_MS
        ]

        instrumental_clip = instrumental[
            :MAX_CLIP_MS
        ]

    length = min(
        len(vocal_clip),
        len(instrumental_clip)
    )

    vocal_clip = vocal_clip[
        :length
    ]

    instrumental_clip = instrumental_clip[
        :length
    ]

    vocal_clip = normalize(
        vocal_clip
    )

    instrumental_clip = normalize(
        instrumental_clip
    )

    # Keep instrumental under the vocal.
    instrumental_clip = (
        instrumental_clip
        + (20 * 0.0)
    )

    instrumental_clip = instrumental_clip.apply_gain(
        -9
    )

    vocal_clip = vocal_clip.apply_gain(
        0
    )

    # Mix vocal + instrumental
    mixed = instrumental_clip.overlay(
        vocal_clip
    )

    fade = min(
        300,
        len(mixed) // 4
    )

    if fade > 0:

        mixed = mixed.fade_in(
            fade
        )

        mixed = mixed.fade_out(
            fade
        )

    return mixed


def create_mashup(clips):

    if not clips:
        return AudioSegment.empty()

    result = clips[0]

    for clip in clips[1:]:

        crossfade = min(
            CROSSFADE_MS,
            len(result) // 3,
            len(clip) // 3
        )

        if crossfade < 200:

            result += clip

        else:

            result = result.append(
                clip,
                crossfade=crossfade
            )

    return result


@app.get("/")
def home():

    return {
        "status": "online",
        "service": "AI Mashup Maker",
        "version": "12.0",
        "ai_model": AI_MODEL,
        "whisper_model": WHISPER_MODEL,
        "demucs": True,
        "device": "cpu",
        "vocal_instrumental_mix": True,
        "crossfade_ms": CROSSFADE_MS
    }


@app.post("/create-mashup")
async def create_mashup(
    files: list[UploadFile] = File(...)
):

    if len(files) < 2:

        return {
            "status": "error",
            "message": "Please upload at least 2 songs."
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
                f"/tmp/"
                f"{job_id}_"
                f"{index}"
                f"{extension}"
            )

            data = await upload.read()

            if len(data) > 25 * 1024 * 1024:

                return {
                    "status": "error",
                    "message":
                        f"{upload.filename} is larger than 25 MB."
                }

            with open(
                input_path,
                "wb"
            ) as f:

                f.write(data)

            # 1. Transcription
            segments = transcribe_audio(
                client,
                input_path
            )

            # 2. Natural lyric lines
            lines = build_lines(
                segments
            )

            # 3. AI selection
            best = choose_best_line(
                client,
                lines
            )

            # 4. Demucs separation
            vocals_path, instrumental_path = (
                separate_audio(
                    input_path,
                    f"{job_id}_{index}"
                )
            )

            # 5. Vocal + instrumental mix
            clip = make_stem_clip(
                vocals_path,
                instrumental_path,
                best
            )

            clips.append(
                clip
            )

            if best:

                selected.append({
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

                selected.append({
                    "song":
                        upload.filename,

                    "text":
                        "",
                    "duration":
                        round(
                            len(clip) / 1000,
                            2
                        )
                })

            if os.path.exists(
                input_path
            ):

                os.remove(
                    input_path
                )

        # 6. Final mashup
        mashup = create_mashup(
            clips
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{job_id}.mp3"
        )

        mashup.export(
            output_path,
            format="mp3",
            bitrate="192k"
        )

        # Cleanup Demucs files
        demucs_job_dir = os.path.join(
            SEPARATION_DIR,
            job_id
        )

        if os.path.exists(
            demucs_job_dir
        ):

            shutil.rmtree(
                demucs_job_dir,
                ignore_errors=True
            )

        return {

            "status":
                "success",

            "job_id":
                job_id,

            "message":
                "AI Vocal + Instrumental Mashup created successfully.",

            "selected_lines":
                selected,

            "audio_info": {

                "demucs":
                    True,

                "vocal_instrumental_mix":
                    True,

                "crossfade_ms":
                    CROSSFADE_MS,

                "duration_seconds":
                    round(
                        len(mashup) / 1000,
                        2
                    )
            },

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
