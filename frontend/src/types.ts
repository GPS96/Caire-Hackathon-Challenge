export type AlertLevel = "Info" | "Caution" | "Emergency";

export interface WaveformSample {
  timestamp: string;
  value: number;
}

export interface InferResponsePayload {
  heart_rate_bpm: number;
  ibi_ms: number[];
  arrhythmia_state: string;
  confidence: number;
  signal_quality: number;
  status: AlertLevel;
  waveform: WaveformSample[];
}

export interface ArrhythmiaPushPayload {
  arrhythmia_state: string;
  confidence: number;
  heart_rate_bpm: number;
  status: AlertLevel;
  generated_at: string;
  // Optional enriched fields from backend websocket (simulation and future real-time)
  signal_quality?: number;
  ibi_ms?: number[];
  waveform?: WaveformSample[];
}

export interface WeeklySessionRow {
  session_id: string;
  started_at: string;
  ended_at: string;
  duration_seconds: number;
  mean_signal_quality: number;
  mean_confidence: number;
  mean_ibi_ms: number;
  mean_heart_rate: number;
  dominant_arrhythmia: string;
  score: number;
}

export interface WeeklySummaryPayload {
  window_start: string;
  window_end: string;
  session_count: number;
  total_drive_time_seconds: number;
  average_signal_quality: number;
  average_confidence: number;
  average_ibi_ms: number;
  average_heart_rate: number;
  average_score: number;
  top_arrhythmia: string | null;
  sessions: WeeklySessionRow[];
}