import os
import re
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field
from youtube_transcript_api import YouTubeTranscriptApi
import openai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title="YouTube Intelligence API",
    description="Comprehensive API for YouTube video analysis and content processing.",
    version="3.0.0",
    contact={"name": "API Support", "email": "support@ytintel.com"},
    license_info={"name": "MIT"},
)

# Models
class TranscriptRequest(BaseModel):
    youtube_url: HttpUrl
    language: Optional[str] = Field("en")

class BatchRequest(BaseModel):
    youtube_urls: List[HttpUrl]
    language: Optional[str] = Field("en")

class TranslateRequest(TranscriptRequest):
    target_language: str

class QuizRequest(TranscriptRequest):
    num_questions: Optional[int] = Field(5, ge=1, le=20)

# Utilities
def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r'(?:v=|/)([0-9A-Za-z_-]{11})',
        r'youtu\.be/([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id: str, language: str = "en") -> str:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript([language])
        data = transcript.fetch()

        lines = []
        for entry in data:
            if isinstance(entry, dict):
                lines.append(entry.get("text", ""))
            else:
                lines.append(getattr(entry, "text", ""))
        return " ".join(lines)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Transcript error: {e}")

def ask_gpt(prompt: str, system: str = "You are a helpful assistant.", max_tokens: int = 400) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

# Health
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/", tags=["Health"])
def root():
    return {"message": "Welcome to YouTube Intelligence API"}

# /video/
@app.post("/video/transcript", tags=["Video"])
def video_transcript(req: TranscriptRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    return {"transcript": get_transcript(vid, req.language)}

@app.post("/video/summary", tags=["Video"])
def video_summary(req: TranscriptRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    t = get_transcript(vid, req.language)[:15000]
    return {"summary": ask_gpt(f"Summarize this video transcript in 3–5 sentences:\n{t}")}

@app.post("/video/categories", tags=["Video"])
def video_categories(req: TranscriptRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    t = get_transcript(vid, req.language)[:15000]
    return {"categories": ask_gpt(f"Categorize this video into 3-5 relevant topics:\n{t}")}

@app.post("/video/full-analysis", tags=["Video"])
def video_full(req: TranscriptRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    t = get_transcript(vid, req.language)[:15000]
    summary = ask_gpt(f"Summarize this:\n{t}")
    cats = ask_gpt(f"Categories:\n{t[:1000]}")
    return {"transcript": t, "summary": summary, "categories": cats}

@app.post("/video/translate", tags=["Video"])
def video_translate(req: TranslateRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    t = get_transcript(vid, req.language)[:15000]
    return {"translated": ask_gpt(f"Translate to {req.target_language}:\n{t}")}

@app.post("/video/entities", tags=["Video"])
def video_entities(req: TranscriptRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    t = get_transcript(vid, req.language)[:15000]
    return {"entities": ask_gpt(f"Extract entities from this transcript:\n{t}")}

@app.post("/video/speakers", tags=["Video"])
def video_speakers(_: TranscriptRequest):
    return {"speakers": ["Speaker 1", "Speaker 2"]}

@app.post("/video/sentiment", tags=["Video"])
def video_sentiment(req: TranscriptRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    t = get_transcript(vid, req.language)[:15000]
    return {"sentiment": ask_gpt(f"Analyze sentiment of this transcript:\n{t}")}

@app.post("/video/chapters", tags=["Video"])
def video_chapters(req: TranscriptRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    t = get_transcript(vid, req.language)[:15000]
    return {"chapters": ask_gpt(f"Generate chapters with timestamps:\n{t}").splitlines()}

@app.post("/video/quiz", tags=["Video"])
def video_quiz(req: QuizRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    t = get_transcript(vid, req.language)[:15000]
    return {"quiz": ask_gpt(f"Create {req.num_questions} quiz questions from this transcript:\n{t}").splitlines()}

@app.post("/video/key-terms", tags=["Video"])
def video_terms(req: TranscriptRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    t = get_transcript(vid, req.language)[:15000]
    return {"terms": ask_gpt(f"Extract key terms:\n{t}")}

@app.post("/video/action-items", tags=["Video"])
def video_actions(req: TranscriptRequest):
    vid = extract_video_id(str(req.youtube_url))
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL format.")
    t = get_transcript(vid, req.language)[:15000]
    return {"actions": ask_gpt(f"What are the actionable items from this video:\n{t}").splitlines()}

# /batch/
@app.post("/batch/translate", tags=["Batch"])
def batch_translate(req: BatchRequest, target_language: str = Query(...)):
    results = []
    for url in req.youtube_urls:
        vid = extract_video_id(str(url))
        if not vid:
            results.append("Error: Invalid YouTube URL format.")
            continue
        try:
            t = get_transcript(vid, req.language)[:15000]
            results.append(ask_gpt(f"Translate to {target_language}:\n{t}"))
        except Exception as e:
            results.append(f"Error: {e}")
    return {"translations": results}

@app.post("/batch/summary", tags=["Batch"])
def batch_summary(req: BatchRequest):
    results = []
    for url in req.youtube_urls:
        vid = extract_video_id(str(url))
        if not vid:
            results.append("Error: Invalid YouTube URL format.")
            continue
        try:
            t = get_transcript(vid, req.language)[:15000]
            results.append(ask_gpt(f"Summarize this transcript:\n{t}"))
        except Exception as e:
            results.append(f"Error: {e}")
    return {"summaries": results}

@app.post("/batch/entities", tags=["Batch"])
def batch_entities(req: BatchRequest):
    results = []
    for url in req.youtube_urls:
        vid = extract_video_id(str(url))
        if not vid:
            results.append("Error: Invalid YouTube URL format.")
            continue
        try:
            t = get_transcript(vid, req.language)[:15000]
            results.append(ask_gpt(f"Extract named entities from:\n{t}"))
        except Exception as e:
            results.append(f"Error: {e}")
    return {"entities": results}

# Global Exception Handler
@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error.", "error": str(exc)})
