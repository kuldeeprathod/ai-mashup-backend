import os
import uuid
import json
import re
import shutil
import subprocess
import threading
import html

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from pydub import AudioSegment
from groq import Groq


app = FastAPI(
    title="AI Mashup Maker",
    version="15.0"
)


OUTPUT_DIR = "/tmp/mashup_outputs"
JOB_DIR = "/tmp/mashup_jobs"
DEMUCS_DIR = "/tmp/demucs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JOB_DIR, exist_ok=True)
os.makedirs(DEMUCS_DIR, exist_ok=True)


AI_MODEL = "openai/gpt-oss-120b"
WHISPER_MODEL = "whisper-large-v3-turbo"

CROSSFADE_MS = 1500
TARGET_DBFS = -16.0

MAX_FILE_MB = 25
MAX_CLIP_MS = 15000


# =========================================================
# HOME
# =========================================================

@app.get("/")
async def home():

    return {
        "status": "online",
        "service": "AI Mashup Maker",
        "version": "15.0",
        "demucs": True,
        "background_processing": True
    }


# =========================================================
# MASHUP WEB APP
# =========================================================

@app.get("/app", response_class=HTMLResponse)
async def mashup_app():

    return """
<!DOCTYPE html>
<html lang="hi">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width,initial-scale=1">

<title>AI Mashup Maker</title>

<style>

*{
    box-sizing:border-box;
}

body{
    margin:0;
    padding:20px;
    font-family:Arial,sans-serif;
    background:#101014;
    color:white;
}

.container{
    max-width:700px;
    margin:auto;
}

.card{
    background:#1b1b22;
    border-radius:18px;
    padding:22px;
    box-shadow:0 10px 40px rgba(0,0,0,.35);
}

h1{
    text-align:center;
    margin-top:0;
}

.subtitle{
    text-align:center;
    color:#aaa;
    margin-bottom:25px;
}

input[type=file]{
    width:100%;
    padding:15px;
    background:#292932;
    color:white;
    border-radius:10px;
    border:1px solid #444;
}

button{
    width:100%;
    margin-top:18px;
    padding:15px;
    border:0;
    border-radius:10px;
    font-size:17px;
    font-weight:bold;
    cursor:pointer;
    background:#ff3158;
    color:white;
}

button:disabled{
    opacity:.5;
    cursor:not-allowed;
}

.progress{
    margin-top:25px;
    display:none;
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

.result{
    display:none;
    margin-top:25px;
}

audio{
    width:100%;
}

.download{
    display:block;
    text-align:center;
    margin-top:15px;
    padding:14px;
    background:#22c55e;
    color:white;
    text-decoration:none;
    border-radius:10px;
    font-weight:bold;
}

.error{
    display:none;
    margin-top:20px;
    padding:15px;
    border-radius:10px;
    background:#4b1515;
    color:#ffb4b4;
    white-space:pre-wrap;
}

.info{
    margin-top:20px;
    padding:14px;
    background:#25252e;
    border-radius:10px;
    color:#bbb;
    font-size:14px;
}

</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>🎵 AI Mashup Maker</h1>

<div class="subtitle">
AI Best Lyrics + Vocal Separation + Instrumental Mixing
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

<div class="status"
     id="status">
Songs processing...
</div>

<div class="bar-bg">

<div
    class="bar"
    id="bar">
</div>

</div>

<div
    id="percent"
    style="margin-top:8px;text-align:center;"
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
    controls>
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
    class="error">
</div>


<div class="info">

<b>कैसे काम करता है:</b><br><br>

🎤 AI lyrics detect करता है<br>
✂️ Best lyric section चुनता है<br>
🎙️ Demucs vocals अलग करता है<br>
🎹 Instrumental अलग करता है<br>
🎚️ Vocal + instrumental mix करता है<br>
🎵 Smooth crossfade लगाता है

</div>

</div>

</div>


<script>

let currentJob = null;


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
        "⏳ Starting...";


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
                "/create-mashup",
                {
                    method:"POST",
                    body:form
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


        setProgress(
            5,
            "AI processing शुरू हो गई..."
        );


        checkJob();


    }catch(error){

        console.error(error);

        showError(
            "Mashup failed:\\n" +
            error.message
        );

        button.disabled =
            false;

        button.innerText =
            "🎧 Create AI Mashup";

    }

}


async function checkJob(){

    try{

        const response =
            await fetch(
                "/status/" +
                encodeURIComponent(
                    currentJob
                )
            );


        if(
            !response.ok
        ){

            throw new Error(
                "Status request failed."
            );

        }


        const data =
            await response.json();


        if(
            data.status ===
            "queued"
        ){

            setProgress(
                5,
                "Mashup queue में है..."
            );

        }


        else if(
            data.status ===
            "processing"
        ){

            setProgress(
                data.progress || 10,
                data.message ||
                "Songs process हो रहे हैं..."
            );

        }


        else if(
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


        else if(
            data.status ===
            "failed"
        ){

            throw new Error(
                data.message ||
                "Processing failed."
            );

        }


        setTimeout(
            checkJob,
            5000
        );


    }catch(error){

        showError(
            "Mashup failed:\\n" +
            error.message
        );

        document.getElementById(
            "start"
        ).disabled =
            false;

        document.getElementById(
            "start"
        ).innerText =
            "🎧 Create AI Mashup";

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

    document.getElementById(
        "start"
    ).disabled =
        false;

    document.getElementById(
        "start"
    ).innerText =
        "🎧 Create AI Mashup";

}


</script>

</body>

</html>
"""


# =========================================================
# GROQ
# =========================================================

def get_client():

    key = os.environ.get(
        "GROQ_API_KEY"
    )

    if not key:

        raise Exception(
            "GROQ_API_KEY is missing."
        )

    return Groq(
        api_key=key
    )


# =========================================================
# TEXT
# =========================================================

def clean_text(text):

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

                language="hi",

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

            text = clean_text(text)

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
# LINES
# =========================================================

def build_lines(
    segments
):

    lines = []

    current = None

    for segment in segments:

        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

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
# AI SELECTION
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

        score = 0

        if 5 <= duration <= 15:

            score += 40

        elif 15 < duration <= 22:

            score += 25

        words = line[
            "text"
        ].split()

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
You are a professional Hindi music mashup editor.

Select ONE strongest lyric section.

Prefer:
- catchy
- emotional
- memorable
- complete phrase
- 5 to 15 seconds
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
# AUDIO
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
# DEMUCS
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
            "Demucs separation failed:\n"
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

    return vocals, instrumental


# =========================================================
# MIX
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
# STATUS
# =========================================================

def save_status(
    job_id,
    data
):

    path = os.path.join(

        JOB_DIR,

        f"{job_id}.json"
    )

    temp = path + ".tmp"

    with open(
        temp,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False
        )

    os.replace(
        temp,
        path
    )


# =========================================================
# BACKGROUND JOB
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

        selected = []

        total = len(
            file_paths
        )

        for index, input_path in enumerate(
            file_paths
        ):

            number = index + 1

            save_status(

                job_id,

                {

                    "status":
                        "processing",

                    "progress":
                        int(
                            index
                            / total
                            * 90
                        ),

                    "message":
                        f"Song {number} analyze हो रहा है..."
                }
            )

            segments = transcribe_audio(
                client,
                input_path
            )

            lines = build_lines(
                segments
            )

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
                                + 0.3
                            )
                            / total
                            * 90
                        ),

                    "message":
                        f"Song {number} की best line चुनी जा रही है..."
                }
            )

            vocals, instrumental = (
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
                                index
                                + 0.8
                            )
                            / total
                            * 90
                        ),

                    "message":
                        f"Song {number} के vocals और instrumental अलग किए जा रहे हैं..."
                }
            )

            clip = create_vocal_mix(

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

        for index, upload in enumerate(
            files
        ):

            extension = os.path.splitext(

                upload.filename
                or ".mp3"

            )[1]

            path = os.path.join(

                folder,

                f"song_{index}"
                f"{extension}"
            )

            data = await upload.read()

            if (
                len(data)
                > MAX_FILE_MB
                * 1024
                * 1024
            ):

                shutil.rmtree(
                    folder,
                    ignore_errors=True
                )

                return {

                    "status":
                        "error",

                    "message":
                        f"{upload.filename} 25 MB से बड़ा है।"
                }

            with open(
                path,
                "wb"
            ) as f:

                f.write(data)

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

        thread = threading.Thread(

            target=
                process_job,

            args=(
                job_id,
                paths
            ),

            daemon=True
        )

        thread.start()

        return {

            "status":
                "queued",

            "job_id":
                job_id
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
async def get_status(
    job_id: str
):

    path = os.path.join(

        JOB_DIR,

        f"{job_id}.json"
    )

    if not os.path.exists(
        path
    ):

        return {

            "status":
                "not_found"
        }

    with open(
        path,
        "r"
    ) as f:

        return json.load(
            f
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
