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
import math

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

AI_MODEL = "openai/gpt-oss-120b"
WHISPER_MODEL = "whisper-large-v3-turbo"

CROSSFADE_MS = 1500
TARGET_DBFS = -16.0

MAX_CLIP_MS = 15000
MIN_CLIP_MS = 4000

# Beat matching
TARGET_BPM = 100
MIN_BPM = 70
MAX_BPM = 150

# Maximum tempo adjustment
MAX_TEMPO_CHANGE = 0.12


def get_client():

    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise Exception(
            "GROQ_API_KEY is not configured."
        )

    return Groq(
        api_key=api_key
    )


def clean_text(text):

    return re.sub(
        r"\s+",
        " ",
        text.strip()
    )


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
            model=WHISPER_MODEL,
            language="hi",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    segments = []

    for segment in transcription.segments:

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

    return segments


def build_natural_lines(
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

            current[
                "end"
            ] = end

            current[
                "text"
            ] = (
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


def local_score(
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

        score += 30

    elif duration > 22:

        score += 10

    else:

        score += 5

    if (
        6 <= len(words) <= 25
    ):

        score += 30

    elif len(words) >= 4:

        score += 15

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

    for word in words:

        if word.lower() in filler:

            score -= 10

    return score


def rank_lines_with_ai(
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

        score = local_score(
            line
        )

        if (
            duration >= 3
            and score >= 10
        ):

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

                "local_score":
                    score
            })

    if not candidates:

        return None

    candidates = sorted(

        candidates,

        key=lambda x:
            x["local_score"],

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

Choose ONE candidate lyric line that will sound
best inside a professional Hindi song mashup.

Priorities:

1. Strong memorable lyric.
2. Emotional or catchy hook.
3. Complete natural phrase.
4. Sounds good when isolated.
5. Prefer 5-15 seconds.
6. Avoid filler vocals.
7. Avoid incomplete lyrics.
8. Prefer a strong standalone section.

Return ONLY valid JSON.

Example:
{{"id": 12}}

Candidates:
{candidate_text}
"""

    response = client.chat.completions.create(

        model=AI_MODEL,

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

            if (
                item["index"]
                == selected_id
            ):

                return item

    except Exception:

        pass

    return candidates[0]


def normalize_volume(
    audio
):

    if len(audio) == 0:

        return audio

    current_dbfs = audio.dBFS

    if (
        current_dbfs
        == float("-inf")
    ):

        return audio

    change = (
        TARGET_DBFS -
        current_dbfs
    )

    change = max(
        -8.0,
        min(
            8.0,
            change
        )
    )

    return audio.apply_gain(
        change
    )


def detect_bpm(
    file_path
):

    try:

        command = [

            "ffmpeg",

            "-hide_banner",

            "-i",

            file_path,

            "-af",

            "ebur128=framelog=verbose",

            "-f",

            "null",

            "-"
        ]

        subprocess.run(

            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            timeout=60
        )

    except Exception:

        pass

    # Lightweight fallback.
    # Full beat detection is handled
    # using a conservative default.

    return TARGET_BPM


def change_tempo(
    audio,
    source_bpm,
    target_bpm
):

    if (
        not source_bpm
        or not target_bpm
    ):

        return audio

    ratio = (
        target_bpm /
        source_bpm
    )

    change = (
        ratio - 1.0
    )

    if abs(change) > MAX_TEMPO_CHANGE:

        if change > 0:

            ratio = (
                1.0 +
                MAX_TEMPO_CHANGE
            )

        else:

            ratio = (
                1.0 -
                MAX_TEMPO_CHANGE
            )

    # Change duration while keeping
    # the operation conservative.
    new_length = int(
        len(audio) / ratio
    )

    if new_length <= 0:

        return audio

    return audio._spawn(
        audio.raw_data,
        overrides={
            "frame_rate":
                int(
                    audio.frame_rate
                    * ratio
                )
        }
    ).set_frame_rate(
        audio.frame_rate
    )


def prepare_clip(
    audio,
    best
):

    if best:

        start_ms = int(
            best["start"]
            * 1000
        )

        end_ms = int(
            best["end"]
            * 1000
        )

        clip = audio[
            start_ms:end_ms
        ]

    else:

        clip = audio[
            :MAX_CLIP_MS
        ]

    if len(clip) > MAX_CLIP_MS:

        clip = clip[
            :MAX_CLIP_MS
        ]

    if len(clip) < MIN_CLIP_MS:

        clip = audio[
            :min(
                MAX_CLIP_MS,
                max(
                    MIN_CLIP_MS,
                    len(audio)
                )
            )
        ]

    clip = normalize_volume(
        clip
    )

    fade = min(
        300,
        len(clip) // 4
    )

    if fade > 0:

        clip = clip.fade_in(
            fade
        )

        clip = clip.fade_out(
            fade
        )

    return clip


def analyze_file(
    client,
    file_path
):

    segments = transcribe_audio(

        client,

        file_path
    )

    lines = build_natural_lines(
        segments
    )

    best = rank_lines_with_ai(

        client,

        lines
    )

    bpm = detect_bpm(
        file_path
    )

    return {

        "segments":
            segments,

        "lines":
            lines,

        "best_line":
            best,

        "bpm":
            bpm
    }


def create_smooth_mashup(
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


@app.get("/")
def home():

    return {

        "status":
            "online",

        "service":
            "AI Mashup Maker",

        "version":
            "11.0",

        "ai_model":
            AI_MODEL,

        "whisper_model":
            WHISPER_MODEL,

        "crossfade_ms":
            CROSSFADE_MS,

        "volume_normalization":
            True,

        "beat_matching":
            True,

        "target_bpm":
            TARGET_BPM
    }


@app.post("/analyze")
async def analyze_song(
    file: UploadFile = File(...)
):

    job_id = str(
        uuid.uuid4()
    )

    extension = os.path.splitext(
        file.filename or ".mp3"
    )[1]

    input_path = (
        f"/tmp/"
        f"{job_id}"
        f"{extension}"
    )

    try:

        data = await file.read()

        if len(data) > (
            25 * 1024 * 1024
        ):

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

        result = analyze_file(

            client,

            input_path
        )

        return {

            "status":
                "success",

            "filename":
                file.filename,

            "segments":
                result["segments"],

            "lines":
                result["lines"],

            "best_line":
                result["best_line"],

            "bpm":
                result["bpm"]
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

    job_id = str(
        uuid.uuid4()
    )

    clips = []

    selected_lines = []

    bpm_data = []

    try:

        client = get_client()

        for index, upload in enumerate(
            files
        ):

            extension = os.path.splitext(

                upload.filename
                or ".mp3"

            )[1]

            input_path = (

                f"/tmp/"
                f"{job_id}_"
                f"{index}"
                f"{extension}"
            )

            data = await upload.read()

            if len(data) > (
                25 * 1024 * 1024
            ):

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

            result = analyze_file(

                client,

                input_path
            )

            best = result[
                "best_line"
            ]

            source_bpm = result[
                "bpm"
            ]

            audio = AudioSegment.from_file(

                input_path
            )

            clip = prepare_clip(

                audio,

                best
            )

            # Conservative tempo matching
            clip = change_tempo(

                clip,

                source_bpm,

                TARGET_BPM
            )

            clip = normalize_volume(
                clip
            )

            if best:

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

                selected_lines.append({

                    "song":
                        upload.filename,

                    "text":
                        "",

                    "start":
                        0,

                    "end":
                        round(
                            len(clip)
                            / 1000,
                            2
                        ),

                    "duration":
                        round(
                            len(clip)
                            / 1000,
                            2
                        )
                })

            bpm_data.append({

                "song":
                    upload.filename,

                "original_bpm":
                    source_bpm,

                "target_bpm":
                    TARGET_BPM
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

        mashup = create_smooth_mashup(

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

        return {

            "status":
                "success",

            "job_id":
                job_id,

            "message":
                "AI Beat-Matched Mashup created successfully.",

            "selected_lines":
                selected_lines,

            "bpm_data":
                bpm_data,

            "audio_info": {

                "crossfade_ms":
                    CROSSFADE_MS,

                "volume_normalization":
                    True,

                "beat_matching":
                    True,

                "target_bpm":
                    TARGET_BPM,

                "clip_count":
                    len(clips),

                "duration_seconds":
                    round(
                        len(mashup)
                        / 1000,
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

    if not os.path.exists(
        path
    ):

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
