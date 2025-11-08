"""Pydantic models for request and response payloads."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    frame_base64: str = Field(..., description="Base64-encoded RGB frame from the in-cabin camera")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SignalSample(BaseModel):
    timestamp: datetime
    value: float


class InferResponse(BaseModel):
    heart_rate_bpm: float = Field(..., ge=0)
    ibi_ms: List[float] = Field(default_factory=list)
    arrhythmia_state: str = Field(..., description="Predicted arrhythmia class from SiamAF")
    confidence: float = Field(..., ge=0.0, le=1.0)
    signal_quality: float = Field(..., ge=0.0, le=1.0)
    status: str = Field(..., description="High-level status banner")
    waveform: List[SignalSample] = Field(default_factory=list, description="Latest samples of the cleaned pulse signal")


class HealthResponse(BaseModel):
    status: str
    backend: str
    ml_pipeline: str
    model_loaded: bool
    device: str


class ArrhythmiaPush(BaseModel):
    arrhythmia_state: str
    confidence: float
    heart_rate_bpm: float
    status: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    # Optional enriched fields (present in simulation and future real-time WS)
    signal_quality: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ibi_ms: List[float] = Field(default_factory=list)
    waveform: List[SignalSample] = Field(default_factory=list)


class SessionReportPayload(BaseModel):
    session_id: Optional[str] = None
    started_at: datetime
    ended_at: datetime
    duration_seconds: float = Field(..., ge=0.0)
    mean_signal_quality: float = Field(..., ge=0.0)
    mean_confidence: float = Field(..., ge=0.0)
    mean_ibi_ms: float = Field(..., ge=0.0)
    mean_heart_rate: float = Field(..., ge=0.0)
    dominant_arrhythmia: str
    score: float = Field(..., ge=0.0, le=100.0)


class SessionReportRow(BaseModel):
    session_id: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float = Field(..., ge=0.0)
    mean_signal_quality: float = Field(..., ge=0.0)
    mean_confidence: float = Field(..., ge=0.0)
    mean_ibi_ms: float = Field(..., ge=0.0)
    mean_heart_rate: float = Field(..., ge=0.0)
    dominant_arrhythmia: str
    score: float = Field(..., ge=0.0, le=100.0)


class WeeklySummaryResponse(BaseModel):
    window_start: datetime
    window_end: datetime
    session_count: int
    total_drive_time_seconds: float = Field(..., ge=0.0)
    average_signal_quality: float = Field(..., ge=0.0)
    average_confidence: float = Field(..., ge=0.0)
    average_ibi_ms: float = Field(..., ge=0.0)
    average_heart_rate: float = Field(..., ge=0.0)
    average_score: float = Field(..., ge=0.0, le=100.0)
    top_arrhythmia: Optional[str] = None
    sessions: List[SessionReportRow] = Field(default_factory=list)
