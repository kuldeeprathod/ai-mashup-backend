```python
import os
import uuid
import json
import re
import shutil
import subprocess
import threading
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from pydub import AudioSegment
from groq import Groq


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="AI Mashup Maker",
    version="16.0"
)


# =========================================================
# DIRECTORIES
# =========================================================

BASE_DIR = "/tmp/ai_mashup"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

JOB_DIR = os.path.join(
    BASE_DIR,
    "jobs"
)

DEMUCS_DIR = os.path.join(
    BASE_DIR,
    "demucs"
)


for folder in [
    BASE_DIR,
    OUTPUT_DIR,
    JOB_DIR,
    DEMUCS_DIR
]:
    os.makedirs(
        folder,
        exist_ok=True
    )


# =========================================================
# CONFIG
# =========================================================

AI_MODEL = "openai/gpt-oss-120b"

WHISPER_MODEL = "whisper-large-v3-turbo"

CROSSFADE_MS = 1500

TARGET_DBFS = -16.0

MAX_FILE_MB = 25

MAX_CLIP_MS = 15000


# =========================================================
# JOB MEMORY
# =========================================================

JOB_STATUS = {}

JOB_LOCK = threading.Lock()


# =========================================================
# ROOT
# =========================================================

@app.get("/")
async def home():

    return {
        "status": "online",
        "service": "AI Mashup Maker",
        "version": "16.0",
        "ai_model": AI_MODEL,
        "whisper_model": WHISPER_MODEL,
        "demucs": True,
        "device": "cpu",
        "vocal_instrumental_mix": True,
        "crossfade_ms": CROSSFADE_MS
    }


# =========================================================
# WEB APP
# =========================================================

@app.get(
    "/app",
    response_class=HTMLResponse
)
async def app_page():

    return """
<!DOCTYPE html>

<html lang="hi">

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width,initial-scale=1"
>

<title>AI Mashup Maker</title>

<style>

*{
    box-sizing:border-box;
}

body{
    margin:0;
    padding:20px;
    background:#101014;
    color:#fff;
    font-family:Arial,sans-serif;
}

.container{
    max-width:700px;
    margin:auto;
}

.card{
    background:#1c1c23;
    padding:24px;
    border-radius:20px;
    box-shadow:
        0 10px 40px
        rgba(0,0,0,.4);
}

h1{
    text-align:center;
    margin:0 0 8px;
}

.subtitle{
    text-align:center;
    color:#aaa;
    margin-bottom:25px;
}

input[type=file]{
    width:100%;
    padding:15px;
    border-radius:12px;
    border:1px solid #444;
    background:#292932;
    color:white;
}

button{
    width:100%;
    margin-top:18px;
    padding:16px;
    border:0;
    border-radius:12px;
    background:#ff3158;
    color:white;
    font-size:17px;
    font-weight:bold;
}

button:disabled{
    opacity:.5;
}

.progress{
    display:none;
    margin-top:25px;
}

.bar-bg{
    width:100%;
    height:14px;
    background:#333;
    border-radius:20px;
    overflow:hidden;
}

.bar{
    width:0%;
    height:100%;
    background:#ff3158;
    transition:width .5s;
}

.status{
    margin:12px 0;
    color:#ddd;
}

.percent{
    text-align:center;
    margin-top:8px;
}

.result{
    display:none;
    margin-top:25px;
}

audio{
    width:100%;
}

.download{
    display:block;
    margin-top:15px;
    padding:15px;
    border-radius:12px;
    text-align:center;
    text-decoration:none;
    background:#22c55e;
    color:white;
    font-weight:bold;
}

.error{
    display:none;
    margin-top:20px;
    padding:15px;
    border-radius:12px;
    background:#4a1515;
    color:#ffb4b4;
    white-space:pre-wrap;
    word-break:break-word;
}

.info{
    margin-top:20px;
    padding:15px;
    background:#27272f;
    border-radius:12px;
    color:#aaa;
    line-height:1.6;
}

</style>

</head>


<body>

<div class="container">

<div class="card">

<h1>🎵 AI Mashup Maker</h1>

<div class="subtitle">
AI Lyrics + Demucs Vocal Separation + Instrumental Mixing
</div>


<input
    id="files"
    type="file"
    accept="audio/*"
    multiple
>


<button
    id="start"
    onclick="startMashup()"
>
🎧 Create AI Mashup
</button>


<div
    id="progress"
    class="progress"
>

<div
    id="status"
    class="status"
>
Starting...
</div>

<div class="bar-bg">

<div
    id="bar"
    class="bar"
>
</div>

</div>

<div
    id="percent"
    class="percent"
>
0%
</div>

</div>


<div
    id="result"
    class="result"
>

<h3>✅ Mashup Ready</h3>

<audio
    id="audio"
    controls
>
</audio>

<a
    id="download"
    class="download"
    download="AI-Mashup.mp3"
>
⬇️ Download Mashup
</a>

</div>


<div
    id="error"
    class="error"
>
</div>


<div class="info">

<b>Processing:</b><br>

🎤 Whisper lyrics detect करता है<br>
🤖 AI best line select करता है<br>
🎙️ Demucs vocals अलग करता है<br>
🎹 Instrumental अलग करता है<br>
🎚️ Vocal + instrumental mix होता है<br>
🎵 Smooth crossfade लगाया जाता है

</div>

</div>

</div>


<script>

let currentJob = null;

let retryCount = 0;


function setProgress(
    value,
    message
){

    value = Math.max(
        0,
        Math.min(
            100,
            value
        )
    );


    document.getElementById(
        "bar"
    ).style.width =
        value + "%";


    document.getElementById(
        "percent"
    ).innerText =
        Math.round(value) + "%";


    document.getElementById(
        "status"
    ).innerText =
        message;
}


function showError(
    message
){

    const box =
        document.getElementById(
            "error"
        );


    box.style.display =
        "block";


    box.innerText =
        message;
}


function resetButton(){

    const button =
        document.getElementById(
            "start"
        );


    button.disabled =
        false;


    button.innerText =
        "🎧 Create AI Mashup";
}


async function startMashup(){

    const input =
        document.getElementById(
            "files"
        );


    const button =
        document.getElementById(
            "start"
        );


    const files =
        input.files;


    document.getElementById(
        "error"
    ).style.display =
        "none";


    document.getElementById(
        "result"
    ).style.display =
        "none";


    if(
        !files ||
        files.length < 2
    ){

        showError(
            "कम से कम 2 songs upload करें।"
        );

        return;
    }


    button.disabled =
        true;


    button.innerText =
        "⏳ Uploading...";


    document.getElementById(
        "progress"
    ).style.display =
        "block";


    setProgress(
        2,
        "Songs upload हो रहे हैं..."
    );


    try{

        const form =
            new FormData();


        for(
            let i = 0;
            i < files.length;
            i++
        ){

            form.append(
                "files",
                files[i]
            );

        }


        const response =
            await fetch(
                window.location.origin +
                "/create-mashup",
                {
                    method:"POST",
                    body:form,
                    cache:"no-store"
                }
            );


        if(
            !response.ok
        ){

            const text =
                await response.text();


            throw new Error(
                "Server error " +
                response.status +
                "\\n" +
                text
            );
        }


        const data =
            await response.json();


        if(
            data.status !==
            "queued"
        ){

            throw new Error(
                data.message ||
                "Mashup start नहीं हुआ।"
            );
        }


        currentJob =
            data.job_id;


        retryCount =
            0;


        setProgress(
            5,
            "AI processing शुरू हो गई..."
        );


        checkJob();


    }catch(error){

        console.error(
            error
        );


        showError(
            "Mashup failed:\\n" +
            error.message
        );


        resetButton();
    }
}


async function checkJob(){

    if(
        !currentJob
    ){

        showError(
            "Job ID नहीं मिला।"
        );

        resetButton();

        return;
    }


    try{

        const url =
            window.location.origin +
            "/status/" +
            encodeURIComponent(
                currentJob
            );


        const response =
            await fetch(
                url,
                {
                    method:"GET",
                    cache:"no-store",
                    headers:{
                        "Accept":
                            "application/json"
                    }
                }
            );


        if(
            !response.ok
        ){

            retryCount++;


            setProgress(
                5,
                "Server response का इंतजार..."
            );


            setTimeout(
                checkJob,
                5000
            );


            return;
        }


        const data =
            await response.json();


        retryCount =
            0;


        console.log(
            "JOB:",
            data
        );


        if(
            data.status ===
            "queued"
        ){

            setProgress(
                data.progress || 5,
                data.message ||
                "Mashup queue में है..."
            );


            setTimeout(
                checkJob,
                5000
            );


            return;
        }


        if(
            data.status ===
            "processing"
        ){

            setProgress(
                data.progress || 10,
                data.message ||
                "Songs process हो रहे हैं..."
            );


            setTimeout(
                checkJob,
                5000
            );


            return;
        }


        if(
            data.status ===
            "completed"
        ){

            setProgress(
                100,
                "🎉 Mashup तैयार है!"
            );


            showResult(
                data
            );


            return;
        }


        if(
            data.status ===
            "failed"
        ){

            showError(
                "Mashup processing failed:\\n\\n" +
                (
                    data.message ||
                    "Unknown processing error."
                )
            );


            resetButton();


            return;
        }


        setProgress(
            5,
            "Processing status check हो रहा है..."
        );


        setTimeout(
            checkJob,
            5000
        );


    }catch(error){

        console.log(
            "Status connection error:",
            error
        );


        retryCount++;


        setProgress(
            5,
            "Connection retry हो रहा है..."
        );


        setTimeout(
            checkJob,
            5000
        );
    }
}


function showResult(
    data
){

    const audio =
        document.getElementById(
            "audio"
        );


    const download =
        document.getElementById(
            "download"
        );


    const url =
        data.download_url;


    audio.src =
        url;


    download.href =
        url;


    document.getElementById(
        "result"
    ).style.display =
        "block";


    resetButton();
}

</script>

</body>

</html>
"""


# =========================================================
# STATUS SAVE
# =========================================================

def save_status(
    job_id,
    data
):

    clean_data = dict(
        data
    )


    with JOB_LOCK:

        JOB_STATUS[
            job_id
        ] = clean_data


    path = os.path.join(
        JOB_DIR,
        f"{job_id}.json"
    )


    try:

        temp =
        path + ".tmp"


        with open(
            temp,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                clean_data,
                f,
                ensure_ascii=False
            )


        os.replace(
            temp,
            path
        )


    except Exception as e:

        print(
            "Status file warning:",
            e
        )


# =========================================================
# GET STATUS
# =========================================================

@app.get(
    "/status/{job_id}"
)
async def get_status(
    job_id: str
):

    with JOB_LOCK:

        data =
            JOB_STATUS.get(
                job_id
            )


    if data:

        return JSONResponse(
            content=data,
            status_code=200
        )


    path = os.path.join(
        JOB_DIR,
        f"{job_id}.json"
    )


    if os.path.exists(path):

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                data =
                    json.load(f)


            with JOB_LOCK:

                JOB_STATUS[
                    job_id
                ] = data


            return JSONResponse(
                content=data,
                status_code=200
            )


        except Exception:

            pass


    return JSONResponse(

        content={

            "status":
                "queued",

            "progress":
                1,

            "message":
                "Job initialize हो रहा है..."
        },

        status_code=200
    )


# =========================================================
# GROQ CLIENT
# =========================================================

def get_client():

    key =
        os.environ.get(
            "GROQ_API_KEY"
        )


    if not key:

        raise Exception(
            "GROQ_API_KEY Render Environment Variables में नहीं मिला।"
        )


    return Groq(
        api_key=key
    )


# =========================================================
# TEXT CLEAN
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

                start =
                    segment["start"]

                end =
                    segment["end"]

                text =
                    segment.get(
                        "text",
                        ""
                    )

            else:

                start =
                    segment.start

                end =
                    segment.end

                text =
                    segment.text


            text =
                clean_text(
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
# BUILD LINES
# =========================================================

def build_lines(
    segments
):

    lines = []

    current = None


    for segment in segments:

        start =
            segment["start"]

        end =
            segment["end"]

        text =
            segment["text"]


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


        gap =
            start -
            current["end"]


        duration =
            current["end"] -
            current["start"]


        if (
            gap <= 0.8
            and duration < 25
        ):

            current["end"] =
                end

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
# AI BEST LINE
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

        duration =
            line["end"] -
            line["start"]


        if duration < 3:

            continue


        score = 0


        if 5 <= duration <= 15:

            score += 40

        elif 15 < duration <= 22:

            score += 25


        words =
            line["text"].split()


        if 6 <= len(words) <= 25:

            score += 30

        elif len(words) >= 4:

            score += 15


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
                score
        })


    if not candidates:

        return None


    candidates.sort(

        key=lambda x:
            x["score"],

        reverse=True
    )


    candidates =
        candidates[:30]


    candidate_text =
        ""


    for item in candidates:

        candidate_text += (

            f'\\nID {item["index"]} | '
            f'{item["duration"]} sec | '
            f'{item["text"]}'
        )


    prompt = f"""
You are a professional Hindi music mashup editor.

Select ONE strongest lyric section.

Prefer:
- catchy
- emotional
- memorable
- complete phrase
- approximately 5 to 15 seconds
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


        match =
            re.search(
                r"\{.*?\}",
                content,
                re.DOTALL
            )


        if match:

            result =
                json.loads(
                    match.group(0)
                )


            selected_id =
                int(
                    result["id"]
                )


            for item in candidates:

                if (
                    item["index"]
                    ==
                    selected_id
                ):

                    return item


    except Exception as e:

        print(
            "AI selection warning:",
            e
        )


    return candidates[0]


# =========================================================
# NORMALIZE
# =========================================================

def normalize_audio(
    audio
):

    if len(audio) == 0:

        return audio


    try:

        db =
            audio.dBFS

    except Exception:

        return audio


    if db == float("-inf"):

        return audio


    gain =
        TARGET_DBFS -
        db


    gain =
        max(
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
# DEMUCS
# =========================================================

def separate_audio(
    input_path,
    job_id
):

    output_dir =
        os.path.join(
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


    process =
        subprocess.run(

            command,

            stdout=
                subprocess.PIPE,

            stderr=
                subprocess.PIPE,

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
            "Demucs separation failed:\n"
            + error[-6000:]
        )


    song_name =
        os.path.splitext(
            os.path.basename(
                input_path
            )
        )[0]


    stem_dir =
        os.path.join(

            output_dir,

            "htdemucs",

            song_name
        )


    vocals =
        os.path.join(
            stem_dir,
            "vocals.wav"
        )


    instrumental =
        os.path.join(
            stem_dir,
            "no_vocals.wav"
        )


    if not os.path.exists(
        vocals
    ):

        raise Exception(
            "Demucs vocals.wav नहीं मिला।"
        )


    if not os.path.exists(
        instrumental
    ):

        raise Exception(
            "Demucs no_vocals.wav नहीं मिला।"
        )


    return (
        vocals,
        instrumental
    )


# =========================================================
# VOCAL + INSTRUMENTAL MIX
# =========================================================

def create_vocal_mix(
    vocals_path,
    instrumental_path,
    best
):

    vocals =
        AudioSegment.from_file(
            vocals_path
        )


    instrumental =
        AudioSegment.from_file(
            instrumental_path
        )


    if best:

        start =
            int(
                best["start"]
                * 1000
            )


        end =
            int(
                best["end"]
                * 1000
            )


        vocal =
            vocals[start:end]


        music =
            instrumental[start:end]


    else:

        vocal =
            vocals[:MAX_CLIP_MS]


        music =
            instrumental[:MAX_CLIP_MS]


    length =
        min(
            len(vocal),
            len(music)
        )


    if length <= 0:

        raise Exception(
            "Audio segment empty है।"
        )


    vocal =
        vocal[:length]


    music =
        music[:length]


    vocal =
        normalize_audio(
            vocal
        )


    music =
        normalize_audio(
            music
        )


    music =
        music.apply_gain(
            -9
        )


    result =
        music.overlay(
            vocal
        )


    fade =
        min(
            300,
            len(result) // 4
        )


    if fade > 0:

        result =
            result.fade_in(
                fade
            )


        result =
            result.fade_out(
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


    result =
        clips[0]


    for clip in clips[1:]:

        crossfade =
            min(

                CROSSFADE_MS,

                len(result) // 3,

                len(clip) // 3
            )


        if crossfade < 200:

            result += clip


        else:

            result =
                result.append(

                    clip,

                    crossfade=
                        crossfade
                )


    return result


# =========================================================
# PROCESS JOB
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


        client =
            get_client()


        clips = []

        selected = []


        total =
            len(
                file_paths
            )


        for index, input_path in enumerate(
            file_paths
        ):

            number =
                index + 1


            save_status(

                job_id,

                {

                    "status":
                        "processing",

                    "progress":
                        int(
                            index /
                            total *
                            85
                        ),

                    "message":
                        f"Song {number} analyze हो रहा है..."
                }
            )


            segments =
                transcribe_audio(
                    client,
                    input_path
                )


            lines =
                build_lines(
                    segments
                )


            best =
                choose_best_line(
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
                                index +
                                0.25
                            )
                            /
                            total *
                            85
                        ),

                    "message":
                        f"Song {number} की best line चुनी जा रही है..."
                }
            )


            vocals, instrumental =
                separate_audio(

                    input_path,

                    f"{job_id}_{index}"
                )


            save_status(

                job_id,

                {

                    "status":
                        "processing",

                    "progress":
                        int(
                            (
                                index +
                                0.75
                            )
                            /
                            total *
                            85
                        ),

                    "message":
                        f"Song {number} के vocals और instrumental अलग किए जा रहे हैं..."
                }
            )


            clip =
                create_vocal_mix(

                    vocals,

                    instrumental,

                    best
                )


            clips.append(
                clip
            )


            if best:

                selected.append({

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
                    "Final mashup बनाया जा रहा है..."
            }
        )


        mashup =
            combine_clips(
                clips
            )


        if len(mashup) == 0:

            raise Exception(
                "Final mashup खाली है।"
            )


        output_path =
            os.path.join(

                OUTPUT_DIR,

                f"{job_id}.mp3"
            )


        mashup.export(

            output_path,

            format="mp3",

            bitrate="192k"
        )


        if not os.path.exists(
            output_path
        ):

            raise Exception(
                "MP3 output create नहीं हुआ।"
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
                    selected,

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


        print(
            "JOB COMPLETED:",
            job_id
        )


    except Exception as e:

        error_text =
            str(e)


        print(
            "JOB FAILED:",
            job_id,
            error_text
        )


        save_status(

            job_id,

            {

                "status":
                    "failed",

                "progress":
                    0,

                "message":
                    error_text
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

        return JSONResponse(

            content={

                "status":
                    "error",

                "message":
                    "कम से कम 2 songs upload करें।"
            },

            status_code=400
        )


    job_id =
        str(
            uuid.uuid4()
        )


    folder =
        os.path.join(

            JOB_DIR,

            job_id
        )


    os.makedirs(
        folder,
        exist_ok=True
    )


    paths = []


    try:

        for index, upload in enumerate(
            files
        ):

            filename =
                upload.filename or "song.mp3"


            extension =
                os.path.splitext(
                    filename
                )[1]


            if not extension:

                extension =
                    ".mp3"


            path =
                os.path.join(

                    folder,

                    f"song_{index}"
                    f"{extension}"
                )


            data =
                await upload.read()


            size_mb =
                len(data) /
                1024 /
                1024


            if size_mb > MAX_FILE_MB:

                shutil.rmtree(
                    folder,
                    ignore_errors=True
                )


                return JSONResponse(

                    content={

                        "status":
                            "error",

                        "message":
                            f"{filename} 25 MB से बड़ा है।"
                    },

                    status_code=400
                )


            with open(
                path,
                "wb"
            ) as f:

                f.write(
                    data
                )


            paths.append(
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


        thread =
            threading.Thread(

                target=
                    process_job,

                args=(
                    job_id,
                    paths
                ),

                daemon=True
            )


        thread.start()


        return JSONResponse(

            content={

                "status":
                    "queued",

                "job_id":
                    job_id,

                "message":
                    "Mashup processing शुरू हो गई।"
            },

            status_code=200
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


        return JSONResponse(

            content={

                "status":
                    "error",

                "message":
                    str(e)
            },

            status_code=500
        )


# =========================================================
# DOWNLOAD
# =========================================================

@app.get(
    "/download/{job_id}"
)
async def download(
    job_id: str
):

    path =
        os.path.join(

            OUTPUT_DIR,

            f"{job_id}.mp3"
        )


    if not os.path.exists(
        path
    ):

        return JSONResponse(

            content={

                "status":
                    "error",

                "message":
                    "Mashup अभी तैयार नहीं है।"
            },

            status_code=404
        )


    return FileResponse(

        path,

        media_type=
            "audio/mpeg",

        filename=
            "AI-Mashup.mp3"
    )
```
