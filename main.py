import os
import re
import json
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import fastapi
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, Field, validator
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from dotenv import load_dotenv
import openai
import uvicorn
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")
openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title="YouTube Summarizer API",
    description="Process and analyze YouTube videos: transcripts, summaries, categories, translation, sentiment, metadata, and more.",
    version="3.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

class VideoRequest(BaseModel):
    youtube_url: HttpUrl

class BatchRequest(BaseModel):
    youtube_urls: List[HttpUrl]

class TranslateRequest(VideoRequest):
    target_language: str

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def fetch_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en'])
        return " ".join([t['text'] for t in transcript.fetch()])
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Transcript error: {e}")

def ask_gpt(prompt: str, system: str = "You are a helpful assistant.", max_tokens: int = 400) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message['content'].strip()

@app.post("/video/transcript")
def get_transcript(request: VideoRequest):
    video_id = extract_video_id(str(request.youtube_url))
    return {"transcript": fetch_transcript(video_id)}

@app.post("/video/summary")
def get_summary(request: VideoRequest):
    transcript = fetch_transcript(extract_video_id(str(request.youtube_url)))
    summary = ask_gpt(f"Summarize this video:\n\n{transcript}")
    return {"summary": summary}

@app.post("/video/categories")
def get_categories(request: VideoRequest):
    transcript = fetch_transcript(extract_video_id(str(request.youtube_url)))
    categories = ask_gpt(
        f"Categorize the content of this transcript into relevant YouTube categories:\n{transcript[:1000]}",
        system="You are an expert at classifying video content."
    )
    return {"categories": re.findall(r'\b[A-Z][a-z]+\b', categories)}

@app.post("/video/full-analysis")
def full_analysis(request: VideoRequest):
    transcript = fetch_transcript(extract_video_id(str(request.youtube_url)))
    summary = ask_gpt(f"Summarize:\n{transcript}")
    categories = ask_gpt(f"What topics?\n{transcript[:1000]}")
    return {
        "transcript": transcript,
        "summary": summary,
        "categories": re.findall(r'\b[A-Z][a-z]+\b', categories)
    }

@app.post("/video/translate")
def translate_video(request: TranslateRequest):
    transcript = fetch_transcript(extract_video_id(str(request.youtube_url)))
    translated = ask_gpt(f"Translate this to {request.target_language}:\n{transcript}")
    return {"translated": translated}

@app.post("/batch/translate")
def batch_translate(request: BatchRequest, target_language: str):
    results = []
    for url in request.youtube_urls:
        try:
            transcript = fetch_transcript(extract_video_id(str(url)))
            translated = ask_gpt(f"Translate to {target_language}:\n{transcript}")
            results.append(translated)
        except Exception as e:
            results.append(f"Error: {e}")
    return {"translations": results}

@app.post("/video/entities")
def extract_entities(request: VideoRequest):
    transcript = fetch_transcript(extract_video_id(str(request.youtube_url)))
    entities = ask_gpt(f"List important names, organizations, or places:\n{transcript[:1500]}")
    return {"entities": re.findall(r'\b[A-Z][a-z]+\b', entities)}

@app.post("/video/speakers")
def detect_speakers(request: VideoRequest):
    return {"speakers": ["Speaker 1", "Speaker 2"]}  # Stubbed

@app.post("/video/sentiment")
def analyze_sentiment(request: VideoRequest):
    transcript = fetch_transcript(extract_video_id(str(request.youtube_url)))
    sentiment = ask_gpt(f"Is the tone positive, neutral, or negative?\n{transcript}")
    return {"sentiment": sentiment}

@app.post("/video/chapters")
def auto_chapters(request: VideoRequest):
    transcript = fetch_transcript(extract_video_id(str(request.youtube_url)))
    chapters = ask_gpt(f"Generate video chapters:\n{transcript}")
    return {"chapters": chapters.splitlines()}

@app.post("/video/quiz")
def generate_quiz(request: VideoRequest, num_questions: Optional[int] = 5):
    transcript = fetch_transcript(extract_video_id(str(request.youtube_url)))
    quiz = ask_gpt(
        f"Create {num_questions} quiz questions about this transcript:\n{transcript[:2000]}"
    )
    return {"quiz": quiz.splitlines()}

@app.post("/video/key-terms")
def key_terms(request: VideoRequest):
    transcript = fetch_transcript(extract_video_id(str(request.youtube_url)))
    terms = ask_gpt(f"Extract key terms from transcript:\n{transcript}")
    return {"terms": re.findall(r'\b\w+\b', terms)}

@app.post("/video/action-items")
def action_items(request: VideoRequest):
    transcript = fetch_transcript(extract_video_id(str(request.youtube_url)))
    actions = ask_gpt(f"Extract action steps from this video:\n{transcript}")
    return {"actions": actions.splitlines()}

@app.post("/batch/summary")
def batch_summary(request: BatchRequest):
    results = []
    for url in request.youtube_urls:
        try:
            transcript = fetch_transcript(extract_video_id(str(url)))
            summary = ask_gpt(f"Summarize:\n{transcript}")
            results.append(summary)
        except Exception as e:
            results.append(f"Error: {e}")
    return {"summaries": results}

@app.post("/batch/entities")
def batch_entities(request: BatchRequest):
    results = []
    for url in request.youtube_urls:
        try:
            transcript = fetch_transcript(extract_video_id(str(url)))
            entities = ask_gpt(f"Entities in video:\n{transcript[:1500]}")
            results.append(re.findall(r'\b[A-Z][a-z]+\b', entities))
        except Exception as e:
            results.append([f"Error: {e}"])
    return {"entities": results}

@app.get("/")
def root():
    return HTMLResponse("""
    <html><body>
    <h2>Welcome to YouTube Summarizer API</h2>
    <p>Visit <a href='/docs'>/docs</a> for the Swagger UI.</p>
    </body></html>
    """)

def extract_video_id(url: str) -> Optional[str]:
    match = re.search(r"(?:v=|youtu.be/)([\w-]{11})", url)
    return match.group(1) if match else None

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
