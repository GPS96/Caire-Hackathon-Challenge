import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useArrhythmiaSocket } from "./hooks/useWebSocket";
import { useEmotion } from "./hooks/useEmotion";
import { useDashboardStore } from "./store/dashboard";
import type { DashboardState, SessionSummary } from "./store/dashboard";
import { EmergencyNavigationFrame } from "./components/EmergencyNavigationFrame";
import { EmergencyActionPrompt } from "./components/EmergencyActionPrompt";
import html2canvas from "html2canvas";
import type { WeeklySummaryPayload } from "./types";

const BACKEND_BASE = __BACKEND_URL__.replace(/\/$/, "");
const REPORT_SESSION_ENDPOINT = "/reports/session";
const REPORT_WEEKLY_ENDPOINT = "/reports/weekly";

const resolveApiUrl = (path: string): string => {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${BACKEND_BASE}${normalized}`;
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [sessionActive, setSessionActive] = useState(false);
  useArrhythmiaSocket(videoRef, sessionActive);

  const heartRate = useDashboardStore((state: DashboardState) => state.heartRate);
  const arrhythmiaState = useDashboardStore((state: DashboardState) => state.arrhythmiaState);
  const status = useDashboardStore((state: DashboardState) => state.status);
  const lastUpdated = useDashboardStore((state: DashboardState) => state.lastUpdated);
  const setMonitoringActive = useDashboardStore((state: DashboardState) => state.setMonitoringActive);
  const sessionSummary = useDashboardStore((state: DashboardState) => state.sessionSummary);
  const finalizeSessionSummary = useDashboardStore((state: DashboardState) => state.finalizeSessionSummary);
  const resetSessionAnalytics = useDashboardStore((state: DashboardState) => state.resetSessionAnalytics);
  const navigationPromptVisible = useDashboardStore((state: DashboardState) => state.navigationPromptVisible);
  const setNavigationVisible = useDashboardStore((state: DashboardState) => state.setNavigationVisible);
  const setNavigationPromptVisible = useDashboardStore((state: DashboardState) => state.setNavigationPromptVisible);
  const silenceNavigation = useDashboardStore((state: DashboardState) => state.silenceNavigation);

  // Removed duplicate declaration of sessionActive and setSessionActive
  const [videoError, setVideoError] = useState<string | null>(null);
  // videoRef already declared above
  const mediaStreamRef = useRef<MediaStream | null>(null);
  // Emotion detection – runs when session is active
  useEmotion(videoRef, sessionActive);

  const [reportSyncError, setReportSyncError] = useState<string | null>(null);
  const [reportSyncRetryKey, setReportSyncRetryKey] = useState(0);
  const [weeklySummaryVisible, setWeeklySummaryVisible] = useState(false);
  const [weeklySummaryLoading, setWeeklySummaryLoading] = useState(false);
  const [weeklySummaryError, setWeeklySummaryError] = useState<string | null>(null);
  const [weeklySummary, setWeeklySummary] = useState<WeeklySummaryPayload | null>(null);
  const uploadedSessionIdsRef = useRef<Set<string>>(new Set());

  const handleEmergencyCall = useCallback(
    (contact: string) => {
      console.log(`Simulated call to ${contact}`);
    },
    []
  );

  const handleWeeklySummaryClick = useCallback(async () => {
    setWeeklySummaryVisible(true);
    setWeeklySummaryLoading(true);
    setWeeklySummaryError(null);
    try {
      const response = await fetch(resolveApiUrl(REPORT_WEEKLY_ENDPOINT), {
        headers: { "Content-Type": "application/json" },
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `HTTP ${response.status}`);
      }
      const data = (await response.json()) as WeeklySummaryPayload;
      setWeeklySummary(data);
    } catch (error: any) {
      const message = error?.message ? String(error.message) : String(error);
      setWeeklySummaryError(message || "Unable to load weekly summary");
      console.error("Failed to fetch weekly summary", error);
    } finally {
      setWeeklySummaryLoading(false);
    }
  }, []);

  const closeWeeklySummary = useCallback(() => {
    setWeeklySummaryVisible(false);
  }, []);

  const retryReportUpload = useCallback(() => {
    setReportSyncError(null);
    setReportSyncRetryKey((key) => key + 1);
  }, []);

  const startSession = useCallback(async () => {
    try {
      setVideoError(null);
      setReportSyncError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setMonitoringActive(true);
      setSessionActive(true);
    } catch (err: any) {
      console.error("Failed to start session (camera)", err);
      setVideoError(err?.message || "Unable to access camera");
      setSessionActive(false);
      setMonitoringActive(false);
    }
  }, [setMonitoringActive, setReportSyncError]);

  // (note) share handled inside DriveSummaryCard

  const stopSession = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    finalizeSessionSummary();
    setSessionActive(false);
    setVideoError(null);
  }, [finalizeSessionSummary]);

  useEffect(() => {
    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!sessionSummary) {
      return;
    }

    const sessionId = sessionSummary.sessionId;
    if (!sessionId || uploadedSessionIdsRef.current.has(sessionId)) {
      return;
    }

    const safeNumber = (value: number): number => (Number.isFinite(value) ? value : 0);
    const startedAtIso = sessionSummary.startedAt ?? sessionSummary.endedAt;
    const endedAtIso = sessionSummary.endedAt;
    if (!endedAtIso) {
      return;
    }

    const startedAt = startedAtIso ? new Date(startedAtIso) : new Date(endedAtIso);
    const endedAt = new Date(endedAtIso);
    const durationSeconds = sessionSummary.durationMs
      ? sessionSummary.durationMs / 1000
      : Math.max(0, (endedAt.getTime() - startedAt.getTime()) / 1000);
    const dominantArrhythmia = sessionSummary.breakdown[0]?.labelKey ?? "unknown";

    const payload = {
      session_id: sessionId,
      started_at: startedAt.toISOString(),
      ended_at: endedAt.toISOString(),
      duration_seconds: safeNumber(durationSeconds),
      mean_signal_quality: safeNumber(sessionSummary.meanSignalQuality),
      mean_confidence: safeNumber(sessionSummary.meanConfidence),
      mean_ibi_ms: safeNumber(sessionSummary.meanIbiMs),
      mean_heart_rate: safeNumber(sessionSummary.meanHeartRate),
      dominant_arrhythmia: dominantArrhythmia,
      score: safeNumber(sessionSummary.score),
    };

    let cancelled = false;
    let retryTimer: number | null = null;
    let activeController: AbortController | null = null;
    const MAX_ATTEMPTS = 3;

    const attemptUpload = async (attempt: number) => {
      const controller = new AbortController();
      activeController = controller;
      try {
        const response = await fetch(resolveApiUrl(REPORT_SESSION_ENDPOINT), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || `HTTP ${response.status}`);
        }

        uploadedSessionIdsRef.current.add(sessionId);
        if (!cancelled) {
          setReportSyncError(null);
        }
      } catch (error: any) {
        if (cancelled || controller.signal.aborted) {
          return;
        }

        const message = error?.message ? String(error.message) : String(error);
        console.error("Failed to persist session report", message);

        if (attempt < MAX_ATTEMPTS) {
          const delay = Math.min(8000, attempt * 2500);
          retryTimer = window.setTimeout(() => {
            attemptUpload(attempt + 1);
          }, delay);
        } else {
          uploadedSessionIdsRef.current.delete(sessionId);
          setReportSyncError("Unable to save drive summary. The system will retry when possible.");
        }
      } finally {
        if (activeController === controller) {
          activeController = null;
        }
      }
    };

    void attemptUpload(1);

    return () => {
      cancelled = true;
      if (retryTimer !== null) {
        window.clearTimeout(retryTimer);
      }
      if (activeController) {
        activeController.abort();
      }
    };
  }, [sessionSummary, reportSyncRetryKey]);

  const formattedHeartRate = Number.isFinite(heartRate) && heartRate > 0 ? `${Math.round(heartRate)}` : "—";
  const formattedArrhythmia = prettifyLabel(arrhythmiaState);
  const severity = useMemo(() => deriveSeverity(arrhythmiaState, status), [arrhythmiaState, status]);
  const centerStatusLabel = sessionActive ? describeSeverity(severity) : "Standby";
  const lastUpdatedLabel = lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : "—";

  return (
    <div className="app-shell">
      {!sessionActive && (
        <div className="session-overlay">
          {sessionSummary ? (
            <DriveSummaryCard
              summary={sessionSummary}
              onStart={startSession}
              onDismiss={resetSessionAnalytics}
            />
          ) : (
            <div className="overlay-card">
              <h1>Guardian Driver Caire</h1>
              <p>Launch a live drive to enable the in-cabin camera, biometric capture, and emergency navigation support.</p>
              {videoError ? <p className="session-error">{videoError}</p> : null}
              <button className="btn btn-primary" type="button" onClick={startSession}>
                Start Drive
              </button>
            </div>
          )}
        </div>
      )}

      <div className="content-wrapper">
        <main className={`drive-layout ${sessionActive ? "" : "layout-muted"}`}>
          <section className="camera-panel">
            <header>
              <h2>In-Cabin Camera</h2>
              <button className="btn btn-ghost" type="button" onClick={stopSession} disabled={!sessionActive}>
                Stop Session
              </button>
            </header>
            <div className="camera-feed">
              <video ref={videoRef} autoPlay muted playsInline />
            </div>
            {videoError ? <p className="camera-error">{videoError}</p> : null}
            <dl className="camera-meta">
              <div>
                <dt>Session</dt>
                <dd>{sessionActive ? "Active" : "Standby"}</dd>
              </div>
              <div>
                <dt>Last Update</dt>
                <dd>{lastUpdatedLabel}</dd>
              </div>
            </dl>
          </section>

          <section className={`guardian-dashboard severity-${severity}`}>
            <div className="dashboard-toolbar">
              <span className="dashboard-title">Guardian Pulse Monitor</span>
              <button
                className="btn btn-ghost"
                type="button"
                onClick={handleWeeklySummaryClick}
                disabled={weeklySummaryLoading}
              >
                {weeklySummaryLoading ? "Loading..." : "Weekly Detail Summary"}
              </button>
            </div>
            {reportSyncError ? (
              <div className="dashboard-inline-alert" role="alert">
                <span>{reportSyncError}</span>
                <button
                  type="button"
                  className="dashboard-inline-alert__retry"
                  onClick={retryReportUpload}
                >
                  Retry now
                </button>
              </div>
            ) : null}
            <div className="pulse-ring">
              <div className="ring-core">
                <span className="status-chip">{formattedArrhythmia}</span>
                <span className="heart-rate-display">
                  {formattedHeartRate}
                  <span className="unit">bpm</span>
                </span>
                <span className="status-label">{centerStatusLabel}</span>
              </div>
            </div>
          </section>
        </main>
      </div>

      <EmergencyNavigationFrame />
      <EmergencyActionPrompt
        visible={navigationPromptVisible}
        onDismiss={() => {
          setNavigationPromptVisible(false);
          silenceNavigation();
        }}
        onNavigate={() => {
          setNavigationPromptVisible(false);
          setNavigationVisible(true);
        }}
        onCall={handleEmergencyCall}
      />
      {weeklySummaryVisible ? (
        <WeeklySummaryDialog
          summary={weeklySummary}
          loading={weeklySummaryLoading}
          error={weeklySummaryError}
          onClose={closeWeeklySummary}
        />
      ) : null}
    </div>
  );
}

type DriveSummaryCardProps = {
  summary: SessionSummary;
  onStart: () => void;
  onDismiss: () => void;
};

function DriveSummaryCard({ summary, onStart, onDismiss }: DriveSummaryCardProps) {
  const clampedScore = Math.max(0, Math.min(100, Math.round(summary.score)));
  const gaugeAngle = clampedScore * 3.6;
  const scoreTone = clampedScore >= 85 ? "stable" : clampedScore >= 65 ? "warning" : "critical";
  const scoreToneMeta = getScoreToneMeta(scoreTone, clampedScore);
  const formattedDuration = formatDuration(summary.durationMs ?? 0);
  const breakdownEntries = summary.breakdown.filter((item) => item.count > 0).sort((a, b) => b.percentage - a.percentage);
  const emotionEntriesAll = (summary.emotionBreakdown || []).filter((i) => i.count > 0).sort((a, b) => b.percentage - a.percentage);
  const topEmotions = emotionEntriesAll.slice(0, 3);
  const topCategories = breakdownEntries.length > 0 ? breakdownEntries.slice(0, 4) : [];
  const cardRef = useRef<HTMLDivElement>(null);
  const [sharing, setSharing] = useState(false);

  const emotionEmojis: Record<string, string> = {
    neutral: "😐",
    happy: "😊",
    sad: "😢",
    angry: "😠",
    fearful: "😨",
    disgusted: "🤢",
    surprised: "😲",
  };

  const handleShare = useCallback(async () => {
    const el = cardRef.current as HTMLElement | null;
    if (!el || sharing) return;
    setSharing(true);
    try {
      // Enter share-mode to disable animations and ensure visibility/opacities
      el.classList.add("share-mode");
      // Safety cleanup timer in case anything throws before our normal cleanup
      const safetyTimer = window.setTimeout(() => el.classList.remove("share-mode"), 8000);
      // Allow styles to apply
      await new Promise((r) => setTimeout(r, 80));
      // Ensure fonts are ready (prevents missing glyphs/boxes)
      // @ts-ignore
      if (document.fonts?.ready) {
        // @ts-ignore
        await document.fonts.ready;
      }

      const canvas = await html2canvas(el, {
        backgroundColor: "#010409",
        scale: Math.max(2, Math.floor(window.devicePixelRatio || 1)),
        logging: false,
        useCORS: true,
        foreignObjectRendering: false,
        imageTimeout: 0,
        removeContainer: true,
        scrollX: 0,
        scrollY: 0,
        onclone: (clonedDoc) => {
          const clonedEl = clonedDoc.querySelector('.summary-card-v2') as HTMLElement | null;
          if (clonedEl) {
            clonedEl.classList.add('share-mode');
            clonedEl.style.maxHeight = 'none';
            clonedEl.style.overflow = 'visible';
            clonedEl.style.width = `${el.scrollWidth}px`;
            clonedEl.style.height = `${el.scrollHeight}px`;
            clonedEl.style.transform = 'none';
            clonedEl.style.position = 'static';
          }
        },
      });
      canvas.toBlob(async (blob) => {
        if (!blob) {
          el.classList.remove("share-mode");
          window.clearTimeout(safetyTimer);
          setSharing(false);
          return;
        }
        const file = new File([blob], "drive-health-summary.png", { type: "image/png" });
        // Try native share API if available
        if (navigator.share && navigator.canShare?.({ files: [file] })) {
          try {
            await navigator.share({ files: [file], title: "Drive Health Summary" });
          } catch (err) {
            // User canceled or share not supported; fallback to download
            downloadBlob(blob, "drive-health-summary.png");
          }
        } else {
          // Fallback: download
          downloadBlob(blob, "drive-health-summary.png");
        }
        el.classList.remove("share-mode");
        window.clearTimeout(safetyTimer);
        setSharing(false);
      }, "image/png");
    } catch (err) {
      console.error("Failed to capture summary:", err);
      el.classList.remove("share-mode");
      // Safety: ensure timer cleared if set
      try { /* noop */ } finally {
        // If a safety timer exists, attempt to clear
      }
      setSharing(false);
    }
  }, [sharing]);

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div ref={cardRef} className="overlay-card summary-card-v2">
      <div className="summary-header-v2">
        <h1>Drive Health Summary</h1>
        <p className="summary-subtitle">
          Guardian monitored <strong>{summary.totalSamples}</strong> rhythm evaluations over <strong>{formattedDuration}</strong>
        </p>
      </div>

      <div className="summary-body-v2">
        {/* Main Guardian Score Ring */}
        <div className="summary-main-ring">
          <div className={`summary-ring-outer summary-ring-outer--${scoreTone}`}>
            <div
              className="summary-ring-fill"
              style={{
                background: `conic-gradient(var(--summary-${scoreTone}-color) 0deg ${gaugeAngle}deg, rgba(15, 23, 42, 0.35) ${gaugeAngle}deg 360deg)`,
              }}
            >
              <div className="summary-ring-inner">
                <span className="summary-ring-score">{clampedScore}</span>
                <span className="summary-ring-label">out of 100</span>
              </div>
            </div>
          </div>
          <div className="summary-ring-caption">
            <span className="summary-ring-status" style={{ color: `var(--summary-${scoreTone}-color)` }}>
              {scoreToneMeta.caption}
            </span>
            <p className="summary-ring-message">{scoreToneMeta.message}</p>
          </div>
        </div>

        {/* Arrhythmia Stats (horizontal ring badges) */}
        <div className="summary-stats-section">
          <h2 className="summary-section-title">Arrhythmia Analysis</h2>
          {topCategories.length === 0 ? (
            <p className="summary-empty">No arrhythmia events detected.</p>
          ) : (
            <div className="summary-ring-badges">
              {topCategories.map((item) => {
                const percent = Math.round(item.percentage);
                const angle = percent * 3.6;
                return (
                  <div key={item.labelKey} className="summary-badge-ring">
                    <div
                      className="badge-ring-outer"
                      style={{
                        background: `conic-gradient(rgba(37, 99, 235, 0.85) 0deg ${angle}deg, rgba(15, 23, 42, 0.3) ${angle}deg 360deg)`,
                      }}
                    >
                      <div className="badge-ring-inner">
                        <span className="badge-ring-value">{percent}</span>
                        <span className="badge-ring-unit">%</span>
                      </div>
                    </div>
                    <div className="badge-ring-info">
                      <span className="badge-ring-label">{item.displayName}</span>
                      <span className="badge-ring-count">{item.count} events</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Emotion Stats (horizontal ring badges with emojis) */}
        {topEmotions.length > 0 && (
          <div className="summary-stats-section">
            <h2 className="summary-section-title">Emotional State</h2>
            <div className="summary-ring-badges">
              {topEmotions.map((item) => {
                const percent = Math.round(item.percentage);
                const angle = percent * 3.6;
                const emoji = emotionEmojis[item.labelKey] || "🙂";
                return (
                  <div key={item.labelKey} className="summary-badge-ring">
                    <div
                      className="badge-ring-outer"
                      style={{
                        background: `conic-gradient(rgba(249, 115, 22, 0.85) 0deg ${angle}deg, rgba(15, 23, 42, 0.3) ${angle}deg 360deg)`,
                      }}
                    >
                      <div className="badge-ring-inner">
                        <span className="badge-ring-emoji">{emoji}</span>
                      </div>
                    </div>
                    <div className="badge-ring-info">
                      <span className="badge-ring-label">{item.displayName}</span>
                      <span className="badge-ring-count">{percent}% • {item.count} samples</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <div className="summary-actions-v2">
        <button className="btn btn-primary btn-large-v2" type="button" onClick={onStart}>
          Start New Drive
        </button>
        <button className="btn btn-secondary btn-large-v2" type="button" onClick={handleShare} disabled={sharing}>
          {sharing ? "Preparing..." : "Share Report"}
        </button>
        <button className="btn btn-ghost btn-large-v2" type="button" onClick={onDismiss}>
          Dismiss Report
        </button>
      </div>
    </div>
  );
}

type WeeklySummaryDialogProps = {
  summary: WeeklySummaryPayload | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
};

function WeeklySummaryDialog({ summary, loading, error, onClose }: WeeklySummaryDialogProps) {
  const sessions = summary?.sessions ?? [];
  const rangeLabel = summary ? formatWeeklyRange(summary.window_start, summary.window_end) : "";
  const totalDriveTime = summary ? formatDurationSeconds(summary.total_drive_time_seconds) : "0m";
  const avgScore = summary && Number.isFinite(summary.average_score) ? Math.round(summary.average_score) : null;
  const avgSignal = summary && Number.isFinite(summary.average_signal_quality) ? Math.round(summary.average_signal_quality * 100) : null;
  const avgConfidence = summary && Number.isFinite(summary.average_confidence) ? Math.round(summary.average_confidence * 100) : null;
  const avgHeartRate = summary && Number.isFinite(summary.average_heart_rate) ? summary.average_heart_rate.toFixed(1) : null;
  const avgIbi = summary && Number.isFinite(summary.average_ibi_ms) ? summary.average_ibi_ms.toFixed(0) : null;
  const topArrhythmia = summary?.top_arrhythmia ? prettifyLabel(summary.top_arrhythmia) : null;

  return (
    <div className="weekly-overlay">
      <div className="overlay-card weekly-summary-card">
        <header className="weekly-header">
          <div>
            <h1>Weekly Detail Summary</h1>
            {rangeLabel ? <p className="weekly-range">{rangeLabel}</p> : null}
          </div>
          <button className="btn btn-ghost" type="button" onClick={onClose}>
            Close
          </button>
        </header>

        {loading ? <p className="weekly-loading">Loading weekly report...</p> : null}

        {error ? (
          <p className="weekly-error" role="alert">
            {error}
          </p>
        ) : null}

        {summary && summary.session_count > 0 ? (
            <>
              <div className="weekly-stats">
                <div>
                  <span className="weekly-stat-label">Sessions</span>
                  <span className="weekly-stat-value">{summary.session_count}</span>
                </div>
                <div>
                  <span className="weekly-stat-label">Drive Time</span>
                  <span className="weekly-stat-value">{totalDriveTime}</span>
                </div>
                <div>
                  <span className="weekly-stat-label">Avg Score</span>
                  <span className="weekly-stat-value">{avgScore !== null ? `${avgScore}` : "—"}</span>
                </div>
                <div>
                  <span className="weekly-stat-label">Avg Signal</span>
                  <span className="weekly-stat-value">{avgSignal !== null ? `${avgSignal}%` : "—"}</span>
                </div>
                <div>
                  <span className="weekly-stat-label">Avg Confidence</span>
                  <span className="weekly-stat-value">{avgConfidence !== null ? `${avgConfidence}%` : "—"}</span>
                </div>
                <div>
                  <span className="weekly-stat-label">Avg Heart Rate</span>
                  <span className="weekly-stat-value">{avgHeartRate !== null ? `${avgHeartRate} bpm` : "—"}</span>
                </div>
                <div>
                  <span className="weekly-stat-label">Avg IBI</span>
                  <span className="weekly-stat-value">{avgIbi !== null ? `${avgIbi} ms` : "—"}</span>
                </div>
                <div>
                  <span className="weekly-stat-label">Top Rhythm</span>
                  <span className="weekly-stat-value">{topArrhythmia ?? "—"}</span>
                </div>
              </div>

              <div className="weekly-session-list">
                <h2>Drive Log</h2>
                <ul>
                  {sessions.map((session) => {
                    const endedAt = new Date(session.ended_at);
                    const endedLabel = Number.isNaN(endedAt.getTime())
                      ? session.ended_at
                      : endedAt.toLocaleString();
                    const duration = formatDurationSeconds(session.duration_seconds);
                    const score = Number.isFinite(session.score) ? Math.round(session.score) : null;
                    const heartRate = Number.isFinite(session.mean_heart_rate)
                      ? `${session.mean_heart_rate.toFixed(1)} bpm`
                      : "—";
                    const signal = Number.isFinite(session.mean_signal_quality)
                      ? `${Math.round(session.mean_signal_quality * 100)}%`
                      : "—";
                    const confidence = Number.isFinite(session.mean_confidence)
                      ? `${Math.round(session.mean_confidence * 100)}%`
                      : "—";
                    const ibi = Number.isFinite(session.mean_ibi_ms) ? `${session.mean_ibi_ms.toFixed(0)} ms` : "—";
                    return (
                      <li key={session.session_id} className="weekly-session-item">
                        <div className="weekly-session-meta">
                          <span className="weekly-session-time">{endedLabel}</span>
                          <span className="weekly-session-duration">{duration}</span>
                        </div>
                        <div className="weekly-session-metrics">
                          <span>Score: {score !== null ? score : "—"}</span>
                          <span>Signal: {signal}</span>
                          <span>Confidence: {confidence}</span>
                          <span>Heart Rate: {heartRate}</span>
                          <span>IBI: {ibi}</span>
                          <span>Dominant: {prettifyLabel(session.dominant_arrhythmia)}</span>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            </>
          ) : !loading && !error ? (
            <p className="weekly-empty">No drives recorded in the last seven days.</p>
          ) : null}
      </div>
    </div>
  );
}

function formatWeeklyRange(startIso: string, endIso: string): string {
  const start = new Date(startIso);
  const end = new Date(endIso);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
    return "";
  }
  const sameYear = start.getFullYear() === end.getFullYear();
  const dateFormatter = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" });
  const yearFormatter = new Intl.DateTimeFormat(undefined, { year: "numeric" });
  const startLabel = dateFormatter.format(start);
  const endLabel = dateFormatter.format(end);
  const startYear = yearFormatter.format(start);
  const endYear = yearFormatter.format(end);
  if (sameYear) {
    return `${startLabel} – ${endLabel}, ${endYear}`;
  }
  return `${startLabel}, ${startYear} – ${endLabel}, ${endYear}`;
}

function formatDurationSeconds(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds <= 0) {
    return "0m";
  }
  const wholeSeconds = Math.round(totalSeconds);
  const hours = Math.floor(wholeSeconds / 3600);
  const minutes = Math.floor((wholeSeconds % 3600) / 60);
  const seconds = wholeSeconds % 60;
  const parts: string[] = [];
  if (hours > 0) {
    parts.push(`${hours}h`);
  }
  if (minutes > 0) {
    parts.push(`${minutes}m`);
  }
  if (parts.length === 0 && seconds > 0) {
    parts.push(`${seconds}s`);
  }
  return parts.join(" ") || "0m";
}

function formatDuration(durationMs: number) {
  if (!durationMs || durationMs < 1000) {
    return "under 1 minute";
  }

  const totalSeconds = Math.round(durationMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const minutePortion = minutes > 0 ? `${minutes} min${minutes === 1 ? "" : "s"}` : "";
  const secondPortion = seconds > 0 ? `${seconds} sec${seconds === 1 ? "" : "s"}` : "";
  return [minutePortion, secondPortion].filter(Boolean).join(" ") || "under 1 minute";
}

function prettifyLabel(label?: string): string {
  if (!label) return "Unknown";
  return label
    .toLowerCase()
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function deriveSeverity(label: string, status: string): "normal" | "caution" | "critical" {
  const normalizedLabel = (label || "").toLowerCase();
  const normalizedStatus = (status || "").toLowerCase();
  if (normalizedLabel.includes("asyst") || normalizedLabel.includes("ventricular") || normalizedStatus === "emergency") {
    return "critical";
  }
  if (
    normalizedLabel.includes("tach") ||
    normalizedLabel.includes("brady") ||
    normalizedLabel.includes("fibrillation") ||
    normalizedStatus === "caution"
  ) {
    return "caution";
  }
  return "normal";
}

function describeSeverity(severity: "normal" | "caution" | "critical"): string {
  switch (severity) {
    case "critical":
      return "Live Critical";
    case "caution":
      return "Live Elevated";
    default:
      return "Live Stable";
  }
}

type ScoreTone = "stable" | "warning" | "critical";

function getScoreToneMeta(tone: ScoreTone, score: number) {
  switch (tone) {
    case "stable":
      return {
        title: "Guarded & Clear",
        message: "Rhythms stayed within expected thresholds.",
        caption: "Recovered Drive",
        icon: (
          <svg width="36" height="36" viewBox="0 0 36 36" fill="none" role="img" aria-label="Stable">
            <circle cx="18" cy="18" r="17" fill="rgba(34,197,94,0.16)" stroke="rgba(34,197,94,0.65)" strokeWidth="2" />
            <path d="M11 18.5l4.8 4.6L25 13" stroke="#4ade80" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ),
      };
    case "warning":
      return {
        title: "Eyes On Recovery",
        message: "Minor arrhythmias surfaced in bursts.",
        caption: "Monitor closely",
        icon: (
          <svg width="36" height="36" viewBox="0 0 36 36" fill="none" role="img" aria-label="Caution">
            <circle cx="18" cy="18" r="17" fill="rgba(250,204,21,0.14)" stroke="rgba(250,204,21,0.62)" strokeWidth="2" />
            <path d="M11 21l4.5-6.5 4.8 5.5L25 12" stroke="#facc15" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M11 24h14" stroke="#facc15" strokeWidth="2.4" strokeLinecap="round" />
          </svg>
        ),
      };
    default:
      return {
        title: "Critical Escalation",
        message: `Severe runs peaked at ${score}% of the drive.`,
        caption: "Escalate to clinician",
        icon: (
          <svg width="36" height="36" viewBox="0 0 36 36" fill="none" role="img" aria-label="Critical">
            <circle cx="18" cy="18" r="17" fill="rgba(239,68,68,0.14)" stroke="rgba(239,68,68,0.68)" strokeWidth="2" />
            <path d="M11 25l7-13 2.5 5L25 10" stroke="#f87171" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M20 12l5 3" stroke="#f87171" strokeWidth="2.4" strokeLinecap="round" />
          </svg>
        ),
      };
  }
}