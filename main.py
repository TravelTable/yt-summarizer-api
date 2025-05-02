import os
import re
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import openai
import uvicorn

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set.")
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")

openai.api_key = OPENAI_API_KEY

app = FastAPI(
    title="YouTube Transcript Summarizer",
    description="Extracts transcript from YouTube, summarizes it, and estimates categories.",
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
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
            except Exception:
                transcript = transcript_list.find_generated_transcript(['en'])
        if not transcript:
            transcript = transcript_list.find_transcript([t.language_code for t in transcript_list])
        transcript_data = transcript.fetch()
        transcript_text = " ".join([entry['text'] for entry in transcript_data])
        return transcript_text
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
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_transcript(transcript: str) -> str:
    if len(transcript.split()) < 1800:
        prompt = (
            "Summarize the following YouTube video transcript in a concise paragraph, "
            "focusing on the main points and key information:\n\n"
            f"{transcript}\n\nSummary:"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.5,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    else:
        chunks = chunk_text(transcript, max_tokens=1800)
        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            prompt = (
                f"Summarize the following part of a YouTube video transcript (part {idx+1}):\n\n"
                f"{chunk}\n\nSummary:"
            )
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.5,
            )
            chunk_summary = response.choices[0].message.content.strip()
            chunk_summaries.append(chunk_summary)
        combined = " ".join(chunk_summaries)
        final_prompt = (
            "Given the following summaries of different parts of a YouTube video transcript, "
            "write a single concise summary paragraph that covers the main points and key information:\n\n"
            f"{combined}\n\nFinal Summary:"
        )
        final_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts."},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=400,
            temperature=0.5,
        )
        final_summary = final_response.choices[0].message.content.strip()
        return final_summary

def categorize_content(transcript: str, summary: str) -> List[str]:
    prompt = (
        "Given the following YouTube video transcript and its summary, "
        "estimate up to 3 most relevant high-level categories for the video. "
        "Choose from categories such as: Education, Technology, Science, Entertainment, Music, News, "
        "Gaming, Sports, How-to, Comedy, Documentary, Health, Business, Finance, Politics, "
        "Travel, Food, Art, History, Lifestyle, Other. "
        "Respond with a JSON list of category names only.\n\n"
        f"Transcript: {transcript[:2000]}...\n\nSummary: {summary}\n\nCategories:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert at classifying YouTube videos by content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=60,
        temperature=0.3,
    )
    categories_text = response.choices[0].message.content.strip()
    try:
        import json
        categories = json.loads(categories_text)
        if isinstance(categories, list):
            return [str(cat) for cat in categories]
    except Exception:
        categories = re.findall(r'"([^"]+)"', categories_text)
        if not categories:
            categories = [cat.strip() for cat in categories_text.split(",") if cat.strip()]
        return categories[:3]
    return categories[:3]

# ---------------------------
# API Endpoints
# ---------------------------

@app.post("/summarize", response_model=TranscriptResponse)
async def summarize_youtube_video(request: TranscriptRequest):
    video_id = extract_video_id(request.youtube_url)
    if not video_id:
        logger.warning(f"Invalid YouTube URL: {request.youtube_url}")
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")
    logger.info(f"Processing video ID: {video_id}")

    transcript = get_transcript(video_id)
    if not transcript or len(transcript.strip()) < 10:
        logger.warning(f"No transcript found for video: {video_id}")
        raise HTTPException(status_code=404, detail="Transcript not available or too short.")

    summary = summarize_transcript(transcript)
    categories = categorize_content(transcript, summary)

    return TranscriptResponse(
        transcript=transcript,
        summary=summary,
        categories=categories
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."}
    )

# ---------------------------
# Main entry point
# ---------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
