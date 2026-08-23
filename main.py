import os
import uuid
import json
import re
import shutil
import subprocess
import threading

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from pydub import AudioSegment
from groq import Groq


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AI Mashup Maker",
    version="14.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],

    allow_headers=["*"],

    expose_headers=["*"],

    max_age=3600
)


# =========================================================
# DIRECTORIES
# =========================================================

OUTPUT_DIR = "/tmp/mashup_outputs"
JOB_DIR = "/tmp/mashup_jobs"
DEMUCS_DIR = "/tmp/demucs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    JOB_DIR,
    exist_ok=True
)

os.makedirs(
    DEMUCS_DIR,
    exist_ok=True
)


# =========================================================
# SETTINGS
# =========================================================

AI_MODEL = "openai/gpt-oss-120b"

WHISPER_MODEL = (
    "whisper-large-v3-turbo"
)

CROSSFADE_MS = 1500

TARGET_DBFS = -16.0

MAX_FILE_MB = 25

MAX_CLIP_MS = 15000

MIN_CLIP_MS = 4000


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def home():

    return {

        "status":
            "online",

        "service":
            "AI Mashup Maker",

        "version":
            "14.0",

        "ai_model":
            AI_MODEL,

        "whisper_model":
            WHISPER_MODEL,

        "demucs":
            True,

        "demucs_device":
            "cpu",

        "background_processing":
            True,

        "vocal_instrumental_mix":
            True,

        "cors":
            True,

        "crossfade_ms":
            CROSSFADE_MS
    }


# =========================================================
# OPTIONS / CORS TEST
# =========================================================

@app.options("/create-mashup")
async def create_mashup_options():

    return JSONResponse(
        content={
            "status": "ok"
        }
    )


# =========================================================
# GROQ CLIENT
# =========================================================

def get_client():

    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise Exception(
            "GROQ_API_KEY is missing."
        )

    return Groq(
        api_key=api_key
    )


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(
    text
):

    return re.sub(
        r"\s+",
        " ",
        str(text).strip()
    )


# =========================================================
# WHISPER
# =========================================================

def transcribe_audio(
    client,
    file_path
):

    with open(
        file_path,
        "rb"
    ) as audio_file:

        result = (
            client.audio
            .transcriptions
            .create(

                file=audio_file,

                model=
                    WHISPER_MODEL,

                language=
                    "hi",

                response_format=
                    "verbose_json",

                timestamp_granularities=[
                    "segment"
                ]
            )
        )

    segments = []

    for segment in result.segments:

        try:

            if isinstance(
                segment,
                dict
            ):

                start = segment[
                    "start"
                ]

                end = segment[
                    "end"
                ]

                text = segment.get(
                    "text",
                    ""
                )

            else:

                start = segment.start

                end = segment.end

                text = segment.text

            text = clean_text(
                text
            )

            if text:

                segments.append({

                    "start":
                        float(start),

                    "end":
                        float(end),

                    "text":
                        text
                })

        except Exception:

            continue

    return segments


# =========================================================
# BUILD NATURAL LINES
# =========================================================

def build_lines(
    segments
):

    lines = []

    current = None

    for segment in segments:

        start = segment[
            "start"
        ]

        end = segment[
            "end"
        ]

        text = segment[
            "text"
        ]

        if current is None:

            current = {

                "start":
                    start,

                "end":
                    end,

                "text":
                    text
            }

            continue

        gap = (
            start -
            current["end"]
        )

        duration = (
            current["end"] -
            current["start"]
        )

        if (
            gap <= 0.8
            and duration < 25
        ):

            current["end"] = end

            current["text"] = (

                current["text"]
                + " "
                + text

            ).strip()

        else:

            lines.append(
                current
            )

            current = {

                "start":
                    start,

                "end":
                    end,

                "text":
                    text
            }

    if current:

        lines.append(
            current
        )

    return lines


# =========================================================
# LOCAL SCORE
# =========================================================

def score_line(
    line
):

    duration = (
        line["end"] -
        line["start"]
    )

    words = line[
        "text"
    ].split()

    score = 0

    if (
        5 <= duration <= 15
    ):

        score += 40

    elif (
        15 < duration <= 22
    ):

        score += 25

    elif duration > 22:

        score += 5

    if (
        6 <= len(words) <= 25
    ):

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


# =========================================================
# AI SELECT BEST LYRIC
# =========================================================

def choose_best_line(
    client,
    lines
):

    if not lines:

        return None

    candidates = []

    for index, line in enumerate(
        lines
    ):

        duration = (

            line["end"] -
            line["start"]

        )

        if duration < 3:

            continue

        candidates.append({

            "index":
                index,

            "start":
                line["start"],

            "end":
                line["end"],

            "duration":
                round(
                    duration,
                    2
                ),

            "text":
                line["text"],

            "score":
                score_line(
                    line
                )
        })

    if not candidates:

        return None

    candidates.sort(

        key=lambda x:
            x["score"],

        reverse=True
    )

    candidates = candidates[
        :30
    ]

    candidate_text = ""

    for item in candidates:

        candidate_text += (

            f'\nID {item["index"]} | '
            f'{item["duration"]} sec | '
            f'{item["text"]}'
        )

    prompt = f"""

You are a professional Hindi
music mashup editor.

Select ONE strongest lyric section.

Prefer:

- catchy
- emotional
- memorable
- complete phrase
- 5 to 15 seconds
- natural singing section
- no filler
- no incomplete lyric

Return ONLY JSON.

Example:

{{"id": 5}}

Candidates:

{candidate_text}

"""

    try:

        response = (
            client.chat
            .completions
            .create(

                model=
                    AI_MODEL,

                temperature=0,

                messages=[

                    {

                        "role":
                            "user",

                        "content":
                            prompt
                    }
                ]
            )
        )

        content = (

            response
            .choices[0]
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

            result = json.loads(
                match.group(0)
            )

            selected_id = int(
                result["id"]
            )

            for item in candidates:

                if (
                    item["index"]
                    == selected_id
                ):

                    return item

    except Exception:

        pass

    return candidates[0]


# =========================================================
# NORMALIZE AUDIO
# =========================================================

def normalize_audio(
    audio
):

    if len(audio) == 0:

        return audio

    if audio.dBFS == float(
        "-inf"
    ):

        return audio

    gain = (

        TARGET_DBFS -
        audio.dBFS
    )

    gain = max(
        -8,
        min(
            8,
            gain
        )
    )

    return audio.apply_gain(
        gain
    )


# =========================================================
# DEMUCS SEPARATION
# =========================================================

def separate_audio(
    input_path,
    job_id
):

    output_dir = os.path.join(

        DEMUCS_DIR,

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

        error = (
            process.stderr
            or process.stdout
            or "Unknown Demucs error."
        )

        raise Exception(
            "Demucs separation failed: "
            + error[-5000:]
        )

    song_name = os.path.splitext(

        os.path.basename(
            input_path
        )

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

    instrumental = os.path.join(

        stem_dir,

        "no_vocals.wav"
    )

    if not os.path.exists(
        vocals
    ):

        raise Exception(
            "Demucs vocals.wav missing."
        )

    if not os.path.exists(
        instrumental
    ):

        raise Exception(
            "Demucs no_vocals.wav missing."
        )

    return (
        vocals,
        instrumental
    )


# =========================================================
# CREATE VOCAL + INSTRUMENTAL MIX
# =========================================================

def create_vocal_mix(
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
            best["start"]
            * 1000
        )

        end = int(
            best["end"]
            * 1000
        )

        vocal = vocals[
            start:end
        ]

        music = instrumental[
            start:end
        ]

    else:

        vocal = vocals[
            :MAX_CLIP_MS
        ]

        music = instrumental[
            :MAX_CLIP_MS
        ]

    length = min(

        len(vocal),

        len(music)
    )

    if length <= 0:

        raise Exception(
            "Empty audio segment."
        )

    vocal = vocal[
        :length
    ]

    music = music[
        :length
    ]

    vocal = normalize_audio(
        vocal
    )

    music = normalize_audio(
        music
    )

    # Instrumental नीचे रहेगा
    music = music.apply_gain(
        -9
    )

    result = music.overlay(
        vocal
    )

    fade = min(

        300,

        len(result) // 4
    )

    if fade > 0:

        result = result.fade_in(
            fade
        )

        result = result.fade_out(
            fade
        )

    return result


# =========================================================
# CROSSFADE
# =========================================================

def combine_clips(
    clips
):

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

                crossfade=
                    crossfade
            )

    return result


# =========================================================
# SAVE JOB STATUS
# =========================================================

def save_status(
    job_id,
    data
):

    status_file = os.path.join(

        JOB_DIR,

        f"{job_id}.json"
    )

    temporary = (
        status_file
        + ".tmp"
    )

    with open(
        temporary,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False
        )

    os.replace(
        temporary,
        status_file
    )


# =========================================================
# BACKGROUND PROCESS
# =========================================================

def process_job(
    job_id,
    file_paths
):

    try:

        save_status(

            job_id,

            {

                "status":
                    "processing",

                "progress":
                    1,

                "message":
                    "AI processing शुरू..."
            }
        )

        client = get_client()

        clips = []

        selected_lines = []

        total = len(
            file_paths
        )

        for index, input_path in enumerate(
            file_paths
        ):

            song_number = (
                index + 1
            )

            save_status(

                job_id,

                {

                    "status":
                        "processing",

                    "progress":
                        int(
                            (
                                index
                                / total
                            ) * 90
                        ),

                    "message":
                        f"Song {song_number} analyze हो रहा है..."
                }
            )

            # Whisper
            segments = transcribe_audio(

                client,

                input_path
            )

            lines = build_lines(
                segments
            )

            # AI lyric
            best = choose_best_line(

                client,

                lines
            )

            save_status(

                job_id,

                {

                    "status":
                        "processing",

                    "progress":
                        int(
                            (
                                index
                                / total
                            ) * 90
                            + 5
                        ),

                    "message":
                        f"Song {song_number} का best line चुना जा रहा है..."
                }
            )

            # Demucs
            vocals_path, instrumental_path = (

                separate_audio(

                    input_path,

                    f"{job_id}_{index}"
                )
            )

            save_status(

                job_id,

                {

                    "status":
                        "processing",

                    "progress":
                        int(
                            (
                                (
                                    index + 0.7
                                )
                                / total
                            ) * 90
                        ),

                    "message":
                        f"Song {song_number} के vocals और instrumental अलग किए जा रहे हैं..."
                }
            )

            # Mix
            clip = create_vocal_mix(

                vocals_path,

                instrumental_path,

                best
            )

            clips.append(
                clip
            )

            if best:

                selected_lines.append({

                    "song":
                        os.path.basename(
                            input_path
                        ),

                    "text":
                        best["text"],

                    "start":
                        best["start"],

                    "end":
                        best["end"],

                    "duration":
                        best["duration"]
                })

        save_status(

            job_id,

            {

                "status":
                    "processing",

                "progress":
                    95,

                "message":
                    "Final mashup तैयार हो रहा है..."
            }
        )

        mashup = combine_clips(
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

        save_status(

            job_id,

            {

                "status":
                    "completed",

                "progress":
                    100,

                "message":
                    "Mashup तैयार है!",

                "selected_lines":
                    selected_lines,

                "duration_seconds":
                    round(
                        len(mashup)
                        / 1000,
                        2
                    ),

                "download_url":
                    f"/download/{job_id}"
            }
        )

    except Exception as e:

        save_status(

            job_id,

            {

                "status":
                    "failed",

                "progress":
                    0,

                "message":
                    str(e)
            }
        )


# =========================================================
# CREATE MASHUP
# =========================================================

@app.post(
    "/create-mashup"
)
async def create_mashup(

    files: list[
        UploadFile
    ] = File(...)
):

    if len(files) < 2:

        return {

            "status":
                "error",

            "message":
                "कम से कम 2 songs upload करें।"
        }

    job_id = str(
        uuid.uuid4()
    )

    job_folder = os.path.join(

        JOB_DIR,

        job_id
    )

    os.makedirs(

        job_folder,

        exist_ok=True
    )

    file_paths = []

    try:

        for index, upload in enumerate(
            files
        ):

            extension = os.path.splitext(

                upload.filename
                or ".mp3"

            )[1]

            path = os.path.join(

                job_folder,

                f"song_{index}"
                f"{extension}"
            )

            data = await upload.read()

            size_mb = (
                len(data)
                / 1024
                / 1024
            )

            if size_mb > MAX_FILE_MB:

                shutil.rmtree(
                    job_folder,
                    ignore_errors=True
                )

                return {

                    "status":
                        "error",

                    "message":
                        f"{upload.filename} "
                        f"25 MB से बड़ा है।"
                }

            with open(
                path,
                "wb"
            ) as f:

                f.write(data)

            file_paths.append(
                path
            )

        save_status(

            job_id,

            {

                "status":
                    "queued",

                "progress":
                    0,

                "message":
                    "Mashup queue में है..."
            }
        )

        worker = threading.Thread(

            target=
                process_job,

            args=(
                job_id,
                file_paths
            ),

            daemon=True
        )

        worker.start()

        return {

            "status":
                "queued",

            "job_id":
                job_id,

            "message":
                "Mashup processing शुरू हो गई।",

            "status_url":
                f"/status/{job_id}"
        }

    except Exception as e:

        return {

            "status":
                "error",

            "message":
                str(e)
        }


# =========================================================
# STATUS
# =========================================================

@app.get(
    "/status/{job_id}"
)
async def job_status(
    job_id: str
):

    status_file = os.path.join(

        JOB_DIR,

        f"{job_id}.json"
    )

    if not os.path.exists(
        status_file
    ):

        return {

            "status":
                "not_found"
        }

    try:

        with open(
            status_file,
            "r"
        ) as f:

            return json.load(
                f
            )

    except Exception as e:

        return {

            "status":
                "error",

            "message":
                str(e)
        }


# =========================================================
# DOWNLOAD
# =========================================================

@app.get(
    "/download/{job_id}"
)
async def download_mashup(
    job_id: str
):

    path = os.path.join(

        OUTPUT_DIR,

        f"{job_id}.mp3"
    )

    if not os.path.exists(
        path
    ):

        return {

            "status":
                "error",

            "message":
                "Mashup अभी तैयार नहीं है।"
        }

    return FileResponse(

        path,

        media_type=
            "audio/mpeg",

        filename=
            "AI-Mashup.mp3"
    )
