# app/models/__init__.py

from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field


class TranscriptSegment(BaseModel):
    """
    Represents a segment of the transcript.
    """
    text: str = Field(..., description="Text content of the transcript segment.")
    start: float = Field(..., description="Start time of the segment in seconds.")
    duration: float = Field(..., description="Duration of the segment in seconds.")


class Transcript(BaseModel):
    """
    Represents the full transcript of a YouTube video.
    """
    video_url: HttpUrl = Field(..., description="URL of the YouTube video.")
    language: Optional[str] = Field(None, description="Language code of the transcript, e.g., 'en'.")
    segments: List[TranscriptSegment] = Field(..., description="List of transcript segments.")
    full_text: str = Field(..., description="Full transcript as a single string.")


class SummaryRequest(BaseModel):
    """
    Request model for summarizing a YouTube video.
    """
    video_url: HttpUrl = Field(..., description="URL of the YouTube video to summarize.")


class SummaryResponse(BaseModel):
    """
    Response model containing the transcript, summary, and estimated categories.
    """
    video_url: HttpUrl = Field(..., description="URL of the YouTube video.")
    transcript: str = Field(..., description="Full transcript of the video.")
    summary: str = Field(..., description="Summarized content of the video.")
    categories: List[str] = Field(..., description="Estimated categories for the video content.")


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    detail: str = Field(..., description="Error message describing what went wrong.")
