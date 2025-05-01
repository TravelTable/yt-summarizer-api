```python
# app/main.py

import os
import re
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, HttpUrl
from fastapi.responses import JSONResponse

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key (ensure this is set in your environment for production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")
openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title="YouTube Transcript Summarizer",
    description="Extracts transcript from YouTube, summarizes it with GPT-4, and estimates categories.",
    version="1.0.0"
)

# ---------------------------
# Pydantic Models
# ---------------------------

class TranscriptRequest(BaseModel):
    youtube_url: HttpUrl

class TranscriptResponse(BaseModel):
    transcript: str
    summary: str
    categories: List[str]

# ---------------------------
# Helper Functions
# ---------------------------

def extract_video_id(youtube_url: str) -> str:
    """
    Extracts the video ID from a YouTube URL.
    """
    # Patterns for various YouTube URL formats
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL format.")

def get_transcript(video_id: str) -> str:
    """
    Retrieves the transcript for a given YouTube video ID.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer English transcript if available
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            # Try manually
            for t in transcript_list:
                if t.language_code.startswith('en'):
                    transcript = t
                    break
            if not transcript:
                transcript = transcript_list.find_manually_created_transcript(transcript_list._langs)
        transcript_data = transcript.fetch()
        full_text = " ".join([entry['text'] for entry in transcript_data if entry['text'].strip()])
        return full_text.strip()
    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript found for this video.")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video unavailable.")
    except Exception as e:
        logger.exception("Error fetching transcript: %s", e)
        raise HTTPException(status_code=500, detail="Error fetching transcript.")

def gpt4_summarize_and_categorize(transcript: str) -> (str, List[str]):
    """
    Uses GPT-4 to summarize the transcript and estimate categories.
    """
    # Truncate transcript if too long for GPT-4 context window
    max_tokens = 7000  # GPT-4 context window is ~8k tokens, but leave room for prompt/response
    # Approximate: 1 token ≈ 4 characters in English
    max_chars = max_tokens * 4
    truncated_transcript = transcript[:max_chars]

    system_prompt = (
        "You are an expert content analyst. "
        "Given a YouTube video transcript, you will:\n"
        "1. Provide a concise summary of the content (max 200 words).\n"
        "2. Estimate up to 5 relevant video categories (as a list of strings, e.g., ['Education', 'Technology']).\n"
        "Respond in JSON with keys 'summary' and 'categories'."
    )

    user_prompt = f"Transcript:\n{truncated_transcript}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,
            max_tokens=800,
        )
        content = response.choices[0].message['content']
        # Try to extract JSON from the response
        import json
        # Sometimes the model may wrap the JSON in markdown code block
        json_match = re.search(r"\{[\s\S]+\}", content)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            summary = result.get("summary", "").strip()
            categories = result.get("categories", [])
            if not isinstance(categories, list):
                categories = [str(categories)]
            return summary, categories
        else:
            # Fallback: try to parse as is
            result = json.loads(content)
            summary = result.get("summary", "").strip()
            categories = result.get("categories", [])
            if not isinstance(categories, list):
                categories = [str(categories)]
            return summary, categories
    except Exception as e:
        logger.exception("Error during GPT-4 summarization/categorization: %s", e)
        raise HTTPException(status_code=500, detail="Error during summarization/categorization.")

# ---------------------------
# API Endpoints
# ---------------------------

@app.post("/summarize", response_model=TranscriptResponse)
async def summarize_transcript(request: TranscriptRequest):
    """
    Given a YouTube URL, extract the transcript, summarize it, and estimate categories.
    """
    try:
        video_id = extract_video_id(request.youtube_url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

    transcript = get_transcript(video_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found or empty.")

    summary, categories = gpt4_summarize_and_categorize(transcript)

    return TranscriptResponse(
        transcript=transcript,
        summary=summary,
        categories=categories
    )

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "YouTube Transcript Summarizer API. POST to /summarize with a YouTube URL."}

# ---------------------------
# Exception Handlers
# ---------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."}
    )
```
