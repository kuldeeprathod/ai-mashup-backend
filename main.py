from fastapi import FastAPI
from groq import Groq
import os

app = FastAPI()


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Groq test server"
    }


@app.get("/test-groq")
def test_groq():

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return {
            "status": "error",
            "message": "GROQ_API_KEY missing"
        }

    try:

        client = Groq(
            api_key=api_key
        )

        models = client.models.list()

        available_models = []

        for model in models.data:

            available_models.append(
                model.id
            )

        return {
            "status": "success",
            "message": "Groq API connected",
            "models": available_models
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }
