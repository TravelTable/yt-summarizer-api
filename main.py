import os
import re
import json
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from dotenv import load_dotenv
import openai
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title="YouTube Summarizer API",
    description="Extracts transcript from YouTube, summarizes it, and estimates categories. Supports sub-endpoints for modular requests.",
    version="2.0.0"
)

# ---------------------------
# Models
# ---------------------------

class VideoRequest(BaseModel):
    youtube_url: HttpUrl

class VideoSummaryResponse(BaseModel):
    transcript: str
    summary: str
    categories: List[str]

class TranscriptResponse(BaseModel):
    transcript: str

class SummaryResponse(BaseModel):
    summary: str

class CategoryResponse(BaseModel):
    categories: List[str]

# ---------------------------
# Helper Functions
# ---------------------------

def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'youtu\.be/([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except Exception:
            transcript = transcript_list.find_generated_transcript(['en'])
        if not transcript:
            transcript = transcript_list.find_transcript([t.language_code for t in transcript_list])
        transcript_data = transcript.fetch()
        return " ".join([entry['text'] for entry in transcript_data])
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        logger.error(f"Transcript error: {e}")
        raise HTTPException(status_code=404, detail="Transcript not available for this video.")
    except Exception as e:
        logger.error(f"Unexpected error fetching transcript: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch transcript.")

def chunk_text(text: str, max_tokens: int = 2000) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_transcript(transcript: str) -> str:
    if len(transcript.split()) < 1800:
        prompt = f"Summarize the following transcript:\n\n{transcript}\n\nSummary:"
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize YouTube videos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    else:
        chunks = chunk_text(transcript, max_tokens=1800)
        summaries = []
        for chunk in chunks:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You summarize YouTube videos."},
                    {"role": "user", "content": f"Summarize:\n\n{chunk}"}
                ],
                max_tokens=400
            )
            summaries.append(response.choices[0].message.content.strip())
        final_summary = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize YouTube videos."},
                {"role": "user", "content": f"Summarize all of this:\n\n{' '.join(summaries)}"}
            ],
            max_tokens=400
        )
        return final_summary.choices[0].message.content.strip()

def categorize_content(transcript: str, summary: str) -> List[str]:
    prompt = (
        "Given the transcript and summary, return 1–3 categories from this list:\n"
        "Education, Technology, Science, Entertainment, Music, News, Gaming, Sports, How-to, Comedy, Documentary, "
        "Health, Business, Finance, Politics, Travel, Food, Art, History, Lifestyle, Other.\n\n"
        f"Transcript: {transcript[:1000]}\n\nSummary: {summary}\n\nCategories:"
    )
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You categorize video content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    try:
        categories = json.loads(response.choices[0].message.content.strip())
        if isinstance(categories, list):
            return categories
    except:
        matches = re.findall(r'"(.*?)"', response.choices[0].message.content)
        return matches or ["Other"]
    return ["Other"]

# ---------------------------
# API Routes
# ---------------------------

@app.post("/video/transcript", response_model=TranscriptResponse)
def get_video_transcript(request: VideoRequest):
    video_id = extract_video_id(str(request.youtube_url))
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
    transcript = get_transcript(video_id)
    return {"transcript": transcript}

@app.post("/video/summary", response_model=SummaryResponse)
def get_video_summary(request: VideoRequest):
    video_id = extract_video_id(str(request.youtube_url))
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
    transcript = get_transcript(video_id)
    summary = summarize_transcript(transcript)
    return {"summary": summary}

@app.post("/video/categories", response_model=CategoryResponse)
def get_video_categories(request: VideoRequest):
    video_id = extract_video_id(str(request.youtube_url))
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
    transcript = get_transcript(video_id)
    summary = summarize_transcript(transcript)
    categories = categorize_content(transcript, summary)
    return {"categories": categories}

@app.post("/batch/full", response_model=VideoSummaryResponse)
def get_full_summary(request: VideoRequest):
    video_id = extract_video_id(str(request.youtube_url))
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
    transcript = get_transcript(video_id)
    summary = summarize_transcript(transcript)
    categories = categorize_content(transcript, summary)
    return {
        "transcript": transcript,
        "summary": summary,
        "categories": categories
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})

# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
