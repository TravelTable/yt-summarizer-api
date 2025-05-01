```python
# app/routers/api_v1.py

from fastapi import APIRouter, HTTPException, status, Body
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import openai
import re
import os

router = APIRouter(
    prefix="/api/v1",
    tags=["YouTube Summarizer"],
)

# Set your OpenAI API key (ensure it's set in your environment for production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set.")
openai.api_key = OPENAI_API_KEY

# ----- Models -----

class SummarizeRequest(BaseModel):
    youtube_url: HttpUrl

class SummarizeResponse(BaseModel):
    transcript: str
    summary: str
    categories: List[str]

# ----- Helper Functions -----

def extract_video_id(youtube_url: str) -> str:
    """
    Extracts the video ID from a YouTube URL.
    Supports various YouTube URL formats.
    """
    # Standard formats
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
    Fetches the transcript for a given YouTube video ID.
    Returns the transcript as a single string.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer English transcript if available
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            # Fallback to manually translated English
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
            except NoTranscriptFound:
                # Fallback to first available transcript
                transcript = transcript_list.find_generated_transcript(transcript_list._langs)
        transcript_data = transcript.fetch()
        full_text = " ".join([entry['text'] for entry in transcript_data])
        return full_text
    except (TranscriptsDisabled, NoTranscriptFound):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not available for this video."
        )
    except VideoUnavailable:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="YouTube video unavailable."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching transcript: {str(e)}"
        )

def gpt4_summarize_and_categorize(transcript: str) -> Dict[str, Any]:
    """
    Uses OpenAI GPT-4 to summarize the transcript and estimate categories.
    Returns a dict with 'summary' and 'categories'.
    """
    # Truncate transcript if too long for context window
    max_tokens = 6000  # GPT-4 context window is ~8k tokens, keep some for prompt/response
    # Approximate: 1 token ≈ 4 chars in English
    max_chars = max_tokens * 4
    transcript_short = transcript[:max_chars]

    prompt = (
        "You are an expert YouTube content analyst. "
        "Given the following transcript of a YouTube video, perform two tasks:\n"
        "1. Write a concise summary (3-6 sentences) of the video's content.\n"
        "2. Estimate 2-4 relevant categories or topics for the video (e.g., 'Technology', 'Education', 'Music', 'Gaming', 'News', etc.). "
        "Return the categories as a Python list of strings.\n\n"
        f"Transcript:\n{transcript_short}\n\n"
        "Respond in the following JSON format:\n"
        "{\n"
        "  \"summary\": \"...\",\n"
        "  \"categories\": [\"...\", \"...\"]\n"
        "}\n"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=512,
        )
        content = response['choices'][0]['message']['content']
        # Try to parse the JSON from the response
        import json
        # Find the JSON object in the response
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in GPT-4 response.")
        json_str = match.group(0)
        result = json.loads(json_str)
        # Validate result
        if 'summary' not in result or 'categories' not in result:
            raise ValueError("Malformed GPT-4 response.")
        if not isinstance(result['categories'], list):
            raise ValueError("Categories must be a list.")
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error summarizing/categorizing with GPT-4: {str(e)}"
        )

# ----- API Route -----

@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize a YouTube video and estimate its categories",
    response_description="The transcript, summary, and estimated categories of the video."
)
async def summarize_youtube_video(request: SummarizeRequest = Body(...)):
    """
    Given a YouTube video URL, extract the transcript, summarize it using GPT-4,
    and estimate relevant categories based on the content.
    """
    try:
        video_id = extract_video_id(request.youtube_url)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    transcript = get_transcript(video_id)
    gpt_result = gpt4_summarize_and_categorize(transcript)

    return SummarizeResponse(
        transcript=transcript,
        summary=gpt_result['summary'],
        categories=gpt_result['categories']
    )
```
