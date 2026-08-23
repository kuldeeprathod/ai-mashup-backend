```python
import os
import uuid
import json
import re
import shutil
import subprocess
import threading

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from pydub import AudioSegment
from groq import Groq


app = FastAPI(
    title="AI Mashup Maker",
    version="17.0"
)


# =========================================================
# CONFIG
# =========================================================

AI_MODEL = "openai/gpt-oss-120b"
WHISPER_MODEL = "whisper-large-v3-turbo"

MAX_FILE_MB = 25
MAX_CLIP_MS = 15000
CROSSFADE_MS = 1500
TARGET_DBFS = -16.0

BASE_DIR = "/tmp/ai_mashup"
JOB_DIR = os.path.join(BASE_DIR, "jobs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DEMUCS_DIR = os.path.join(BASE_DIR, "demucs")

for directory in [BASE_DIR, JOB_DIR, OUTPUT_DIR, DEMUCS_DIR]:
    os.makedirs(directory, exist_ok=True)


# =========================================================
# JOB STORAGE
# =========================================================

JOB_STATUS = {}
JOB_LOCK = threading.Lock()


def save_status(job_id, data):
    with JOB_LOCK:
        JOB_STATUS[job_id] = dict(data)

    path = os.path.join(JOB_DIR, job_id + ".json")
    temp = path + ".tmp"

    try:
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        os.replace(temp, path)

    except Exception as exc:
        print("Status save warning:", exc)


def read_status(job_id):
    with JOB_LOCK:
        if job_id in JOB_STATUS:
            return JOB_STATUS[job_id]

    path = os.path.join(JOB_DIR, job_id + ".json")

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            with JOB_LOCK:
                JOB_STATUS[job_id] = data

            return data

        except Exception:
            pass

    return None


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def home():
    return {
        "status": "online",
        "service": "AI Mashup Maker",
        "version": "17.0",
        "ai_model": AI_MODEL,
        "whisper_model": WHISPER_MODEL,
        "demucs": True,
        "device": "cpu",
        "vocal_instrumental_mix": True,
        "crossfade_ms": CROSSFADE_MS
    }


# =========================================================
# SIMPLE WEB APP
# =========================================================

@app.get("/app", response_class=HTMLResponse)
async def mashup_app():
    return """
<!DOCTYPE html>
<html lang="hi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>AI Mashup Maker</title>

<style>
body {
    margin: 0;
    padding: 20px;
    background: #101014;
    color: white;
    font-family: Arial, sans-serif;
}

.box {
    max-width: 700px;
    margin: auto;
    background: #1c1c23;
    padding: 25px;
    border-radius: 20px;
}

h1 {
    text-align: center;
}

p {
    color: #aaa;
}

input {
    width: 100%;
    padding: 15px;
    margin-top: 15px;
    background: #292932;
    color: white;
    border: 1px solid #444;
    border-radius: 10px;
    box-sizing: border-box;
}

button {
    width: 100%;
    margin-top: 18px;
    padding: 15px;
    border: 0;
    border-radius: 10px;
    background: #ff3158;
    color: white;
    font-size: 17px;
    font-weight: bold;
}

button:disabled {
    opacity: 0.5;
}

#progressBox,
#result,
#error {
    display: none;
    margin-top: 25px;
}

.barBackground {
    width: 100%;
    height: 14px;
    background: #333;
    border-radius: 20px;
    overflow: hidden;
}

#bar {
    width: 0%;
    height: 100%;
    background: #ff3158;
}

audio {
    width: 100%;
    margin-top: 10px;
}

.download {
    display: block;
    margin-top: 15px;
    padding: 15px;
    background: #22c55e;
    color: white;
    text-align: center;
    text-decoration: none;
    border-radius: 10px;
}

#error {
    background: #481515;
    color: #ffb4b4;
    padding: 15px;
    border-radius: 10px;
    white-space: pre-wrap;
}
</style>
</head>

<body>

<div class="box">

<h1>🎵 AI Mashup Maker</h1>

<p>
AI lyrics selection + Demucs vocal separation +
instrumental mixing
</p>

<input
    id="files"
    type="file"
    accept="audio/*"
    multiple
>

<button id="start" onclick="createMashup()">
🎧 Create AI Mashup
</button>

<div id="progressBox">

<p id="status">Starting...</p>

<div class="barBackground">
    <div id="bar"></div>
</div>

<p id="percent">0%</p>

</div>

<div id="error"></div>

<div id="result">

<h3>✅ Mashup Ready</h3>

<audio id="audio" controls></audio>

<a id="download" class="download" download="AI-Mashup.mp3">
⬇️ Download Mashup
</a>

</div>

</div>


<script>

let jobId = null;


function progress(value, text) {

    document.getElementById("bar").style.width =
        value + "%";

    document.getElementById("percent").innerText =
        Math.round(value) + "%";

    document.getElementById("status").innerText =
        text;
}


function errorMessage(text) {

    const box = document.getElementById("error");

    box.style.display = "block";
    box.innerText = text;

    document.getElementById("start").disabled = false;
    document.getElementById("start").innerText =
        "🎧 Create AI Mashup";
}


async function createMashup() {

    const input = document.getElementById("files");
    const files = input.files;

    document.getElementById("error").style.display = "none";
    document.getElementById("result").style.display = "none";
    document.getElementById("progressBox").style.display = "block";

    if (!files || files.length < 2) {

        errorMessage(
            "कम से कम 2 songs upload करें।"
        );

        return;
    }

    const button = document.getElementById("start");

    button.disabled = true;
    button.innerText = "⏳ Uploading...";

    progress(2, "Songs upload हो रहे हैं...");

    try {

        const form = new FormData();

        for (let i = 0; i < files.length; i++) {
            form.append("files", files[i]);
        }

        const response = await fetch(
            "/create-mashup",
            {
                method: "POST",
                body: form
            }
        );

        const data = await response.json();

        if (!response.ok || data.status !== "queued") {

            throw new Error(
                data.message || "Mashup start नहीं हुआ।"
            );
        }

        jobId = data.job_id;

        progress(
            5,
            "AI processing शुरू हो गई..."
        );

        checkStatus();

    } catch (error) {

        errorMessage(
            "Mashup failed:\\n" + error.message
        );
    }
}


async function checkStatus() {

    if (!jobId) {
        errorMessage("Job ID नहीं मिला।");
        return;
    }

    try {

        const response = await fetch(
            "/status/" + encodeURIComponent(jobId),
            {
                cache: "no-store"
            }
        );

        if (!response.ok) {

            setTimeout(checkStatus, 5000);
            return;
        }

        const data = await response.json();

        if (data.status === "queued") {

            progress(
                data.progress || 5,
                data.message || "Queue में है..."
            );

            setTimeout(checkStatus, 5000);
            return;
        }


        if (data.status === "processing") {

            progress(
                data.progress || 10,
                data.message || "Processing..."
            );

            setTimeout(checkStatus, 5000);
            return;
        }


        if (data.status === "completed") {

            progress(
                100,
                "🎉 Mashup तैयार है!"
            );

            document.getElementById("audio").src =
                data.download_url;

            document.getElementById("download").href =
                data.download_url;

            document.getElementById("result").style.display =
                "block";

            buttonReset();

            return;
        }


        if (data.status === "failed") {

            errorMessage(
                "Mashup failed:\\n\\n" +
                (data.message || "Unknown error")
            );

            return;
        }


        setTimeout(checkStatus, 5000);

    } catch (error) {

        progress(
            5,
            "Server connection retry हो रहा है..."
        );

        setTimeout(checkStatus, 5000);
    }
}


function buttonReset() {

    const button = document.getElementById("start");

    button.disabled = false;
    button.innerText = "🎧 Create AI Mashup";
}

</script>

</body>
</html>
"""


# =========================================================
# GROQ
# =========================================================

def get_groq_client():

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY Render Environment Variables में missing है।"
        )

    return Groq(api_key=api_key)


# =========================================================
# WHISPER TRANSCRIPTION
# =========================================================

def transcribe_audio(client, file_path):

    with open(file_path, "rb") as audio_file:

        result = client.audio.transcriptions.create(
            file=audio_file,
            model=WHISPER_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    segments = []

    for segment in result.segments:

        try:

            if isinstance(segment, dict):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "")
            else:
                start = segment.start
                end = segment.end
                text = segment.text

            text = re.sub(
                r"\s+",
                " ",
                str(text).strip()
            )

            if text:
                segments.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "text": text
                    }
                )

        except Exception:
            continue

    return segments


# =========================================================
# BUILD LYRIC LINES
# =========================================================

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

        duration = current["end"] - current["start"]

        if gap <= 0.8 and duration < 25:

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


# =========================================================
# AI BEST LINE
# =========================================================

def choose_best_line(client, lines):

    candidates = []

    for index, line in enumerate(lines):

        duration = (
            line["end"] -
            line["start"]
        )

        if duration < 3:
            continue

        score = 0

        if 5 <= duration <= 15:
            score += 40
        elif duration <= 22:
            score += 20

        word_count = len(
            line["text"].split()
        )

        if 6 <= word_count <= 25:
            score += 30
        elif word_count >= 4:
            score += 15

        candidates.append(
            {
                "id": index,
                "start": line["start"],
                "end": line["end"],
                "duration": round(duration, 2),
                "text": line["text"],
                "score": score
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    candidates = candidates[:30]

    candidate_text = "\n".join(
        [
            "ID {} | {} sec | {}".format(
                item["id"],
                item["duration"],
                item["text"]
            )
            for item in candidates
        ]
    )

    prompt = f"""
You are a professional Hindi music mashup editor.

Select ONE strongest lyric section.

Prefer a catchy, emotional and memorable
complete lyric phrase.

Prefer approximately 5 to 15 seconds.

Return ONLY JSON.

Example:
{{"id": 5}}

Candidates:

{candidate_text}
"""

    try:

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
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        match = re.search(
            r"\{.*?\}",
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

                if item["id"] == selected_id:
                    return item

    except Exception as exc:

        print(
            "AI selection warning:",
            exc
        )

    return candidates[0]


# =========================================================
# DEMUCS
# =========================================================

def separate_audio(input_path, job_id):

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

    print(
        "Running Demucs:",
        " ".join(command)
    )

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800
    )

    if process.returncode != 0:

        error = (
            process.stderr or
            process.stdout or
            "Unknown Demucs error."
        )

        raise RuntimeError(
            "Demucs separation failed:\n" +
            error[-6000:]
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

    instrumental = os.path.join(
        stem_dir,
        "no_vocals.wav"
    )

    if not os.path.exists(vocals):
        raise RuntimeError(
            "Demucs vocals.wav नहीं मिला।"
        )

    if not os.path.exists(instrumental):
        raise RuntimeError(
            "Demucs no_vocals.wav नहीं मिला।"
        )

    return vocals, instrumental


# =========================================================
# AUDIO NORMALIZATION
# =========================================================

def normalize_audio(audio):

    if len(audio) == 0:
        return audio

    try:
        db = audio.dBFS
    except Exception:
        return audio

    if db == float("-inf"):
        return audio

    gain = TARGET_DBFS - db

    gain = max(
        -8,
        min(8, gain)
    )

    return audio.apply_gain(gain)


# =========================================================
# VOCAL + INSTRUMENTAL
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
            best["start"] * 1000
        )

        end = int(
            best["end"] * 1000
        )

        vocal = vocals[start:end]
        music = instrumental[start:end]

    else:

        vocal = vocals[:MAX_CLIP_MS]
        music = instrumental[:MAX_CLIP_MS]

    length = min(
        len(vocal),
        len(music)
    )

    if length <= 0:
        raise RuntimeError(
            "Audio segment empty है।"
        )

    vocal = vocal[:length]
    music = music[:length]

    vocal = normalize_audio(vocal)
    music = normalize_audio(music)

    music = music.apply_gain(-9)

    result = music.overlay(vocal)

    fade = min(
        300,
        len(result) // 4
    )

    if fade > 0:

        result = result.fade_in(fade)
        result = result.fade_out(fade)

    return result


# =========================================================
# COMBINE
# =========================================================

def combine_clips(clips):

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


# =========================================================
# BACKGROUND PROCESS
# =========================================================

def process_job(job_id, file_paths):

    try:

        save_status(
            job_id,
            {
                "status": "processing",
                "progress": 1,
                "message": "AI processing शुरू..."
            }
        )

        client = get_groq_client()

        clips = []
        selected_lines = []

        total = len(file_paths)

        for index, input_path in enumerate(file_paths):

            number = index + 1

            base_progress = int(
                index / total * 90
            )

            save_status(
                job_id,
                {
                    "status": "processing",
                    "progress": base_progress,
                    "message":
                        f"Song {number} analyze हो रहा है..."
                }
            )

            segments = transcribe_audio(
                client,
                input_path
            )

            lines = build_lines(segments)

            best = choose_best_line(
                client,
                lines
            )

            save_status(
                job_id,
                {
                    "status": "processing",
                    "progress":
                        min(base_progress + 10, 90),
                    "message":
                        f"Song {number} की best line चुनी जा रही है..."
                }
            )

            vocals, instrumental = separate_audio(
                input_path,
                job_id + "_" + str(index)
            )

            save_status(
                job_id,
                {
                    "status": "processing",
                    "progress":
                        min(base_progress + 20, 92),
                    "message":
                        f"Song {number} के vocals और instrumental अलग किए जा रहे हैं..."
                }
            )

            clip = create_vocal_mix(
                vocals,
                instrumental,
                best
            )

            clips.append(clip)

            if best:

                selected_lines.append(
                    {
                        "song":
                            os.path.basename(input_path),
                        "text":
                            best["text"],
                        "start":
                            best["start"],
                        "end":
                            best["end"],
                        "duration":
                            best["duration"]
                    }
                )

        save_status(
            job_id,
            {
                "status": "processing",
                "progress": 95,
                "message":
                    "Final mashup बनाया जा रहा है..."
            }
        )

        mashup = combine_clips(clips)

        if len(mashup) == 0:
            raise RuntimeError(
                "Final mashup खाली है।"
            )

        output_path = os.path.join(
            OUTPUT_DIR,
            job_id + ".mp3"
        )

        mashup.export(
            output_path,
            format="mp3",
            bitrate="192k"
        )

        if not os.path.exists(output_path):
            raise RuntimeError(
                "MP3 output create नहीं हुआ।"
            )

        save_status(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "message": "Mashup तैयार है!",
                "selected_lines":
                    selected_lines,
                "duration_seconds":
                    round(len(mashup) / 1000, 2),
                "download_url":
                    "/download/" + job_id
            }
        )

        print(
            "JOB COMPLETED:",
            job_id
        )

    except Exception as exc:

        print(
            "JOB FAILED:",
            job_id,
            str(exc)
        )

        save_status(
            job_id,
            {
                "status": "failed",
                "progress": 0,
                "message": str(exc)
            }
        )


# =========================================================
# CREATE MASHUP
# =========================================================

@app.post("/create-mashup")
async def create_mashup(
    files: list[UploadFile] = File(...)
):

    if len(files) < 2:

        return JSONResponse(
            {
                "status": "error",
                "message":
                    "कम से कम 2 songs upload करें।"
            },
            status_code=400
        )

    job_id = str(uuid.uuid4())

    folder = os.path.join(
        JOB_DIR,
        job_id
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    paths = []

    try:

        for index, upload in enumerate(files):

            filename = (
                upload.filename or
                "song.mp3"
            )

            extension = os.path.splitext(
                filename
            )[1]

            if not extension:
                extension = ".mp3"

            path = os.path.join(
                folder,
                "song_" +
                str(index) +
                extension
            )

            data = await upload.read()

            size_mb = (
                len(data) /
                1024 /
                1024
            )

            if size_mb > MAX_FILE_MB:

                shutil.rmtree(
                    folder,
                    ignore_errors=True
                )

                return JSONResponse(
                    {
                        "status": "error",
                        "message":
                            f"{filename} 25 MB से बड़ा है।"
                    },
                    status_code=400
                )

            with open(path, "wb") as f:
                f.write(data)

            paths.append(path)

        save_status(
            job_id,
            {
                "status": "queued",
                "progress": 0,
                "message":
                    "Mashup queue में है..."
            }
        )

        thread = threading.Thread(
            target=process_job,
            args=(job_id, paths),
            daemon=True
        )

        thread.start()

        return {
            "status": "queued",
            "job_id": job_id,
            "message":
                "Mashup processing शुरू हो गई।"
        }

    except Exception as exc:

        save_status(
            job_id,
            {
                "status": "failed",
                "progress": 0,
                "message": str(exc)
            }
        )

        return JSONResponse(
            {
                "status": "error",
                "message": str(exc)
            },
            status_code=500
        )


# =========================================================
# STATUS
# =========================================================

@app.get("/status/{job_id}")
async def status(job_id: str):

    data = read_status(job_id)

    if data is None:

        return {
            "status": "queued",
            "progress": 1,
            "message":
                "Job initialize हो रहा है..."
        }

    return data


# =========================================================
# DOWNLOAD
# =========================================================

@app.get("/download/{job_id}")
async def download(job_id: str):

    path = os.path.join(
        OUTPUT_DIR,
        job_id + ".mp3"
    )

    if not os.path.exists(path):

        return JSONResponse(
            {
                "status": "error",
                "message":
                    "Mashup अभी तैयार नहीं है।"
            },
            status_code=404
        )

    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename="AI-Mashup.mp3"
    )
```
