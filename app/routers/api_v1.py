from fastapi import APIRouter, HTTPException, status, Body
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import openai
import re
import os
import json

router = APIRouter(
    prefix="/api/v1",
    tags=["YouTube Summarizer"],
)

# Set your OpenAI API key
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
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
            except NoTranscriptFound:
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
    max_tokens = 6000
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
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in GPT-4 response.")
        json_str = match.group(0)
        result = json.loads(json_str)
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
