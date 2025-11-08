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

const resolveApiUrl = (path: string): string => `${BACKEND_BASE}${path.startsWith("/") ? path : `/${path}`}`;

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [sessionActive, setSessionActive] = useState(false);
  useArrhythmiaSocket(videoRef, sessionActive);

  const heartRate = useDashboardStore((s: DashboardState) => s.heartRate);
  const arrhythmiaState = useDashboardStore((s: DashboardState) => s.arrhythmiaState);
  const status = useDashboardStore((s: DashboardState) => s.status);
  const lastUpdated = useDashboardStore((s: DashboardState) => s.lastUpdated);
  const setMonitoringActive = useDashboardStore((s: DashboardState) => s.setMonitoringActive);
  const sessionSummary = useDashboardStore((s: DashboardState) => s.sessionSummary);
  const finalizeSessionSummary = useDashboardStore((s: DashboardState) => s.finalizeSessionSummary);
  const resetSessionAnalytics = useDashboardStore((s: DashboardState) => s.resetSessionAnalytics);
  const navigationPromptVisible = useDashboardStore((s: DashboardState) => s.navigationPromptVisible);
  const setNavigationVisible = useDashboardStore((s: DashboardState) => s.setNavigationVisible);
  const setNavigationPromptVisible = useDashboardStore((s: DashboardState) => s.setNavigationPromptVisible);
  const silenceNavigation = useDashboardStore((s: DashboardState) => s.silenceNavigation);

  const [videoError, setVideoError] = useState<string | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  useEmotion(videoRef, sessionActive);

  const [reportSyncError, setReportSyncError] = useState<string | null>(null);
  const [reportSyncRetryKey, setReportSyncRetryKey] = useState(0);
  const [weeklySummaryVisible, setWeeklySummaryVisible] = useState(false);
  const [weeklySummaryLoading, setWeeklySummaryLoading] = useState(false);
  const [weeklySummaryError, setWeeklySummaryError] = useState<string | null>(null);
  const [weeklySummary, setWeeklySummary] = useState<WeeklySummaryPayload | null>(null);
  const uploadedSessionIdsRef = useRef<Set<string>>(new Set());

  const handleEmergencyCall = useCallback((contact: string) => {
    console.log(`Simulated call to ${contact}`);
  }, []);

  const handleWeeklySummaryClick = useCallback(async () => {
    setWeeklySummaryVisible(true);
    setWeeklySummaryLoading(true);
    setWeeklySummaryError(null);
    try {
      const resp = await fetch(resolveApiUrl(REPORT_WEEKLY_ENDPOINT), { headers: { "Content-Type": "application/json" } });
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(detail || `HTTP ${resp.status}`);
      }
      const data = (await resp.json()) as WeeklySummaryPayload;
      setWeeklySummary(data);
    } catch (e: any) {
      const msg = e?.message ? String(e.message) : String(e);
      setWeeklySummaryError(msg || "Unable to load weekly summary");
      console.error("Failed to fetch weekly summary", e);
    } finally {
      setWeeklySummaryLoading(false);
    }
  }, []);

  const closeWeeklySummary = useCallback(() => setWeeklySummaryVisible(false), []);
  const retryReportUpload = useCallback(() => {
    setReportSyncError(null);
    setReportSyncRetryKey((k) => k + 1);
  }, []);

  const startSession = useCallback(async () => {
    try {
      setVideoError(null);
      setReportSyncError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      mediaStreamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setMonitoringActive(true);
      setSessionActive(true);
    } catch (err: any) {
      console.error("Failed to start session (camera)", err);
      setVideoError(err?.message || "Unable to access camera");
      setSessionActive(false);
      setMonitoringActive(false);
    }
  }, [setMonitoringActive]);

  const stopSession = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    finalizeSessionSummary();
    setSessionActive(false);
    setVideoError(null);
  }, [finalizeSessionSummary]);

  useEffect(() => () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
  }, []);

  // Persist session summary reports with retry logic
  useEffect(() => {
    if (!sessionSummary) return;
    const sessionId = sessionSummary.sessionId;
    if (!sessionId || uploadedSessionIdsRef.current.has(sessionId)) return;
    const safe = (v: number) => (Number.isFinite(v) ? v : 0);
    const startedIso = sessionSummary.startedAt ?? sessionSummary.endedAt;
    const endedIso = sessionSummary.endedAt;
    if (!endedIso) return;
    const startedAt = startedIso ? new Date(startedIso) : new Date(endedIso);
    const endedAt = new Date(endedIso);
    const durationSeconds = sessionSummary.durationMs
      ? sessionSummary.durationMs / 1000
      : Math.max(0, (endedAt.getTime() - startedAt.getTime()) / 1000);
    const dominant = sessionSummary.breakdown[0]?.labelKey ?? "unknown";
    const payload = {
      session_id: sessionId,
      started_at: startedAt.toISOString(),
      ended_at: endedAt.toISOString(),
      duration_seconds: safe(durationSeconds),
      mean_signal_quality: safe(sessionSummary.meanSignalQuality),
      mean_confidence: safe(sessionSummary.meanConfidence),
      mean_ibi_ms: safe(sessionSummary.meanIbiMs),
      mean_heart_rate: safe(sessionSummary.meanHeartRate),
      dominant_arrhythmia: dominant,
      score: safe(sessionSummary.score),
    };
    let cancelled = false;
    let timer: number | null = null;
    let controller: AbortController | null = null;
    const MAX = 3;
    const upload = async (attempt: number) => {
      controller = new AbortController();
      try {
        const resp = await fetch(resolveApiUrl(REPORT_SESSION_ENDPOINT), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        if (!resp.ok) {
          const detail = await resp.text();
          throw new Error(detail || `HTTP ${resp.status}`);
        }
        uploadedSessionIdsRef.current.add(sessionId);
        if (!cancelled) setReportSyncError(null);
      } catch (e: any) {
        if (cancelled || controller.signal.aborted) return;
        if (attempt < MAX) {
          const delay = Math.min(8000, attempt * 2500);
          timer = window.setTimeout(() => upload(attempt + 1), delay);
        } else {
          uploadedSessionIdsRef.current.delete(sessionId);
          setReportSyncError("Unable to save drive summary. The system will retry when possible.");
        }
      }
    };
    upload(1);
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
      if (controller) controller.abort();
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
            <DriveSummaryCard summary={sessionSummary} onStart={startSession} onDismiss={resetSessionAnalytics} />
          ) : (
            <div className="overlay-card">
              <h1>Guardian Driver Caire</h1>
              <p>Launch a live drive to enable the in-cabin camera, biometric capture, and emergency navigation support.</p>
              {videoError ? <p className="session-error">{videoError}</p> : null}
              <button className="btn btn-primary" type="button" onClick={startSession}>Start Drive</button>
            </div>
          )}
        </div>
      )}

      <div className="content-wrapper">
        <main className={`drive-layout ${sessionActive ? "" : "layout-muted"}`}>
          <section className="camera-panel">
            <header>
              <h2>In-Cabin Camera</h2>
              <button className="dashboard-icon-btn icon-power-off" type="button" onClick={stopSession} disabled={!sessionActive} aria-label="Stop Session" title="Stop Session">
                {/* Power Off Icon */}
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M6.2 6.2a8 8 0 1 0 11.3 0" />
                  <path d="M12 2v10" />
                </svg>
              </button>
            </header>
            <div className="camera-feed"><video ref={videoRef} autoPlay muted playsInline /></div>
            {videoError ? <p className="camera-error">{videoError}</p> : null}
            <dl className="camera-meta">
              <div><dt>Session</dt><dd>{sessionActive ? "Active" : "Standby"}</dd></div>
              <div><dt>Last Update</dt><dd>{lastUpdatedLabel}</dd></div>
            </dl>
          </section>

          <section className={`guardian-dashboard severity-${severity}`}>
            <div className="dashboard-toolbar">
              <span className="dashboard-title">Guardian Pulse Monitor</span>
              <button className="dashboard-icon-btn" type="button" onClick={handleWeeklySummaryClick} disabled={weeklySummaryLoading} aria-label="Weekly Detail Summary" title="Weekly Detail Summary">
                {weeklySummaryLoading ? (
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="spin-icon">
                    <circle cx="12" cy="12" r="10" strokeDasharray="60" strokeDashoffset="20" />
                  </svg>
                ) : (
                  /* Analytics bar/line combination icon */
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 3v18h18" />
                    <rect x="6" y="13" width="3" height="5" rx="1" />
                    <rect x="11" y="9" width="3" height="9" rx="1" />
                    <rect x="16" y="5" width="3" height="13" rx="1" />
                    <path d="M6 11l3-2 2 1 5-5 3 2" />
                  </svg>
                )}
              </button>
            </div>
            {reportSyncError ? (
              <div className="dashboard-inline-alert" role="alert">
                <span>{reportSyncError}</span>
                <button type="button" className="dashboard-inline-alert__retry" onClick={retryReportUpload}>Retry now</button>
              </div>
            ) : null}
            <div className="pulse-ring">
              <div className="ring-core">
                <span className="status-chip">{formattedArrhythmia}</span>
                <span className="heart-rate-display">{formattedHeartRate}<span className="unit">bpm</span></span>
                <span className="status-label">{centerStatusLabel}</span>
              </div>
            </div>
          </section>
        </main>
      </div>

      <EmergencyNavigationFrame />
      <EmergencyActionPrompt
        visible={navigationPromptVisible}
        onDismiss={() => { setNavigationPromptVisible(false); silenceNavigation(); }}
        onNavigate={() => { setNavigationPromptVisible(false); setNavigationVisible(true); }}
        onCall={handleEmergencyCall}
      />
      {weeklySummaryVisible && (
        <WeeklySummaryDialog
          summary={weeklySummary}
          loading={weeklySummaryLoading}
          error={weeklySummaryError}
          onClose={closeWeeklySummary}
        />
      )}
    </div>
  );
}

type DriveSummaryCardProps = { summary: SessionSummary; onStart: () => void; onDismiss: () => void };

function DriveSummaryCard({ summary, onStart, onDismiss }: DriveSummaryCardProps) {
  const clampedScore = Math.max(0, Math.min(100, Math.round(summary.score)));
  // Compute arrhythmia vs normal percentages and use the higher one as the center ring value
  const arrEntries = summary.breakdown.filter((i) => i.count > 0).sort((a, b) => b.percentage - a.percentage);
  const arrhythmiaPct = arrEntries.reduce((acc, it) => {
    try { return acc + (isArrhythmiaLabel((it.labelKey || '').toLowerCase()) ? (Number.isFinite(it.percentage) ? it.percentage : 0) : 0); } catch (e) { return acc; }
  }, 0);
  const normalPct = Math.max(0, 100 - arrhythmiaPct);
  const centerValue = Math.max(0, Math.min(100, Math.round(Math.max(arrhythmiaPct, normalPct))));
  const gaugeAngle = centerValue * 3.6;
  // default tone; will be overridden based on dominant class below
  let scoreTone: ScoreTone = "critical";
  const meanFatigueVal = typeof summary.meanFatigue === 'number' ? Math.max(0, Math.min(100, Math.round(summary.meanFatigue))) : null;
  // ringCompareClass will be computed after arrEntries are known (based on Normal vs Arrhythmia dominance)
  let ringCompareClass = '';
  const formattedDuration = formatDuration(summary.durationMs ?? 0);
  const topCategories = arrEntries.slice(0, 4);
  // Compute dominance between Normal and Arrhythmia across breakdown entries
  const dominantClass = arrhythmiaPct > normalPct ? 'Arrhythmia' : 'Normal';
  const dominantPct = Math.round(Math.max(arrhythmiaPct, normalPct));
  ringCompareClass = arrhythmiaPct > normalPct ? 'neo-red' : 'neo-green';
  // Map dominant class to ring tone/colors/messages: Normal -> stable/green, Arrhythmia -> critical/red
  scoreTone = dominantClass === 'Normal' ? 'stable' : 'critical';
  const scoreToneMeta = getScoreToneMeta(scoreTone, centerValue);
  const emotionEntries = (summary.emotionBreakdown || []).filter((e) => e.count > 0).sort((a, b) => b.percentage - a.percentage);
  // If fatigue is present and shown as the first badge, prefer the 2nd+3rd top emotions
  const topEmotions = typeof summary.meanFatigue === 'number' ? emotionEntries.slice(1, 3) : emotionEntries.slice(0, 2);
  const secondEmotion = emotionEntries[1] ?? emotionEntries[0] ?? null;
  const cardRef = useRef<HTMLDivElement>(null);
  const [sharing, setSharing] = useState(false);
  const emotionEmojis: Record<string, string> = { neutral: "😐", happy: "😊", sad: "😢", angry: "😠", fearful: "😨", disgusted: "🤢", surprised: "😲" };

  const handleShare = useCallback(async () => {
    const el = cardRef.current; if (!el || sharing) return; setSharing(true);
    try {
      el.classList.add("share-mode");
      const safety = window.setTimeout(() => el.classList.remove("share-mode"), 8000);
      await new Promise((r) => setTimeout(r, 80));
      // @ts-ignore
      if (document.fonts?.ready) { // @ts-ignore
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
        onclone: (doc) => {
          const cloned = doc.querySelector('.summary-card-v2') as HTMLElement | null;
          if (cloned) {
            cloned.classList.add('share-mode');
            cloned.style.maxHeight = 'none';
            cloned.style.overflow = 'visible';
            cloned.style.width = `${el.scrollWidth}px`;
            cloned.style.height = `${el.scrollHeight}px`;
            cloned.style.transform = 'none';
            cloned.style.position = 'static';
          }
        },
      });
      canvas.toBlob((blob) => {
        if (!blob) { el.classList.remove("share-mode"); window.clearTimeout(safety); setSharing(false); return; }
        const file = new File([blob], "drive-health-summary.png", { type: "image/png" });
        if (navigator.share && navigator.canShare?.({ files: [file] })) {
          navigator.share({ files: [file], title: "Drive Health Summary" }).catch(() => downloadBlob(blob, "drive-health-summary.png"));
        } else {
          downloadBlob(blob, "drive-health-summary.png");
        }
        el.classList.remove("share-mode"); window.clearTimeout(safety); setSharing(false);
      }, "image/png");
    } catch (e) {
      console.error("Failed to capture summary", e); cardRef.current?.classList.remove("share-mode"); setSharing(false);
    }
  }, [sharing]);

  const downloadBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = filename; document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
  };

  return (
    <div ref={cardRef} className="overlay-card summary-card-v2">
      <div className="summary-window-controls">
  <button className="summary-icon-btn icon-power-on" type="button" aria-label="Start New Drive" title="Start New Drive" onClick={onStart}>
          {/* Power Icon */}
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2v10" />
            <path d="M6.2 6.2a8 8 0 1 0 11.3 0" />
          </svg>
        </button>
        <button className="summary-icon-btn" type="button" aria-label="Share Report" title="Share Report" onClick={handleShare} disabled={sharing}>
          {/* Export / Share Icon */}
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.0" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 15V3" />
              <path d="M6.5 8.5 12 3l5.5 5.5" />
              <path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" />
            </svg>
        </button>
        <button className="summary-icon-btn summary-icon-danger" type="button" aria-label="Dismiss Report" title="Dismiss Report" onClick={onDismiss}>
          {/* Close Icon */}
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>
      </div>
      <div className="summary-header-v2">
        <h1>Drive Health Summary</h1>
        <p className="summary-subtitle">Guardian monitored <strong>{summary.totalSamples}</strong> rhythm evaluations over <strong>{formattedDuration}</strong></p>
        {/* Fatigue previously displayed in header; moved to the right column above Emotional State */}
      </div>
      <div className={`summary-body-v2 ${topEmotions.length === 0 ? "summary-body-v2--no-emotions" : ""}`}>
        {/* Left: Arrhythmia Analysis */}
        <div className="summary-stats-section summary-stats-section--left">
          <h2 className="summary-section-title">Arrhythmia Analysis</h2>
          {topCategories.length === 0 ? <p className="summary-empty">No arrhythmia events detected.</p> : (
            <div className="summary-ring-badges">
              {topCategories.map((item) => {
                const percent = Math.round(item.percentage); const angle = percent * 3.6;
                return (
                  <div key={item.labelKey} className="summary-badge-ring">
                    <div className="badge-ring-outer" style={{ background: `conic-gradient(rgba(37,99,235,0.85) 0deg ${angle}deg, rgba(15,23,42,0.3) ${angle}deg 360deg)` }}>
                      <div className="badge-ring-inner">
                        <span className="badge-ring-value">{percent}</span><span className="badge-ring-unit">%</span>
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
        {/* Center: Guardian Score Ring */}
        <div className="summary-main-ring">
          <div className={`summary-ring-outer summary-ring-outer--${scoreTone} ${ringCompareClass}` }>
            <div className="summary-ring-fill" style={{ background: `conic-gradient(var(--summary-${scoreTone}-color) 0deg ${gaugeAngle}deg, rgba(15,23,42,0.35) ${gaugeAngle}deg 360deg)` }}>
              <div className="summary-ring-inner">
                <span className="summary-ring-score">{centerValue}</span>
                <span className="summary-ring-label">out of 100</span>
              </div>
            </div>
          </div>
          <div className="summary-ring-caption">
            <span className="summary-ring-status" style={{ color: `var(--summary-${scoreTone}-color)` }}>{scoreToneMeta.caption}</span>
            <p className="summary-ring-message">{scoreToneMeta.message}</p>
          </div>
        </div>
        {/* Right: Emotional State (fatigue moved here above the emotional stats) */}
        {topEmotions.length > 0 && (
          <div className="summary-stats-section summary-stats-section--right">
            {/* Fatigue small badge (if available) — render as a card-like badge */}
            {meanFatigueVal !== null && (() => {
              const v = meanFatigueVal as number;
              const angle = v * 3.6;
              let color = 'rgba(34,197,94,0.85)';
              if (v >= 70) color = 'rgba(239,68,68,0.9)';
              else if (v >= 40) color = 'rgba(250,204,21,0.92)';
              const pulseClass = v >= 70 ? 'pulse' : '';
              return (
                <div style={{ marginBottom: 12 }}>
                  <div className="summary-badge-ring" style={{ alignItems: 'center', padding: '8px 10px', minWidth: 140 }}>
                    <div className="badge-ring-outer" style={{ width: 52, height: 52, background: `conic-gradient(${color} 0deg ${angle}deg, rgba(15,23,42,0.3) ${angle}deg 360deg)` }}>
                      <div className={`badge-ring-inner ${pulseClass}`} style={{ width: 40, height: 40 }}>
                        <span className="badge-ring-value" style={{ fontSize: 13 }}>{v}</span><span className="badge-ring-unit" style={{ fontSize: 10 }}>%</span>
                      </div>
                    </div>
                    <div className="badge-ring-info" style={{ marginLeft: 12, textAlign: 'left' }}>
                      <span className="badge-ring-label">Fatigue</span>
                      <span className="badge-ring-count">Average • {v}%</span>
                    </div>
                  </div>
                </div>
              );
            })()}
            <h2 className="summary-section-title">Emotional State</h2>
            <div className="summary-ring-badges">
                  {/* Show only a single emotion card (the second-highest) under Emotional Stats */}
                  {secondEmotion ? (
                    (() => {
                      const item = secondEmotion;
                      const percent = Math.round(item.percentage);
                      const angle = percent * 3.6;
                      const emoji = emotionEmojis[item.labelKey] || "🙂";
                      return (
                        <div key={item.labelKey} className="summary-badge-ring">
                          <div className="badge-ring-outer" style={{ background: `conic-gradient(rgba(249,115,22,0.85) 0deg ${angle}deg, rgba(15,23,42,0.3) ${angle}deg 360deg)` }}>
                            <div className="badge-ring-inner"><span className="badge-ring-emoji">{emoji}</span></div>
                          </div>
                          <div className="badge-ring-info">
                            <span className="badge-ring-label">{item.displayName}</span>
                            <span className="badge-ring-count">{percent}% • {item.count} samples</span>
                          </div>
                        </div>
                      );
                    })()
                  ) : (
                    <p className="summary-empty">No emotion samples</p>
                  )}
                </div>
          </div>
        )}
      </div>
      <div className="summary-actions-v2">
        {/* Replaced by icon controls top-right */}
      </div>
    </div>
  );
}

type WeeklySummaryDialogProps = { summary: WeeklySummaryPayload | null; loading: boolean; error: string | null; onClose: () => void };

function WeeklySummaryDialog({ summary, loading, error, onClose }: WeeklySummaryDialogProps) {
  const sessions = summary?.sessions ?? [];
  const rangeLabel = summary ? formatWeeklyRange(summary.window_start, summary.window_end) : "";
  const totalDriveTime = summary ? formatDurationSeconds(summary.total_drive_time_seconds) : "0m";
  const avgScore = summary && Number.isFinite(summary.average_score) ? Math.round(summary.average_score) : null;
  const avgHeartRate = summary && Number.isFinite(summary.average_heart_rate) ? summary.average_heart_rate.toFixed(1) : null;
  const avgIbi = summary && Number.isFinite(summary.average_ibi_ms) ? summary.average_ibi_ms.toFixed(0) : null;
  const topArrhythmia = summary?.top_arrhythmia ? prettifyLabel(summary.top_arrhythmia) : null;
  // Compute comparative normal vs arrhythmia percentages across sessions
  const sessionCount = summary?.session_count ?? 0;
  const arrhythmiaSessionCount = summary
    ? summary.sessions.filter((s) => isArrhythmiaLabel((s.dominant_arrhythmia || "").toLowerCase())).length
    : 0;
  const normalSessionCount = Math.max(0, sessionCount - arrhythmiaSessionCount);
  const arrhythmiaPercent = sessionCount > 0 ? Math.round((arrhythmiaSessionCount / sessionCount) * 100) : 0;
  const normalPercent = sessionCount > 0 ? Math.round((normalSessionCount / sessionCount) * 100) : 0;
  // ref + sharing state for share/export capture
  const weeklyRef = useRef<HTMLDivElement | null>(null);
  const [sharing, setSharing] = useState(false);

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

  const handleWeeklyShare = useCallback(async () => {
    const el = weeklyRef.current;
    if (!el || sharing) return;
    setSharing(true);
    try {
      el.classList.add("share-mode");
      const safety = window.setTimeout(() => el.classList.remove("share-mode"), 8000);
      await new Promise((r) => setTimeout(r, 80));
      // @ts-ignore
      if (document.fonts?.ready) { await document.fonts.ready; }
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
        onclone: (doc) => {
          const cloned = doc.querySelector('.weekly-summary-card') as HTMLElement | null;
          if (cloned) {
            cloned.classList.add('share-mode');
            cloned.style.maxHeight = 'none';
            cloned.style.overflow = 'visible';
            cloned.style.width = `${el.scrollWidth}px`;
            cloned.style.height = `${el.scrollHeight}px`;
            cloned.style.transform = 'none';
            cloned.style.position = 'static';
          }
        },
      });
      canvas.toBlob((blob) => {
        if (!blob) { el.classList.remove("share-mode"); window.clearTimeout(safety); setSharing(false); return; }
        const file = new File([blob], "weekly-drive-summary.png", { type: "image/png" });
        if (navigator.share && navigator.canShare?.({ files: [file] })) {
          navigator.share({ files: [file], title: "Weekly Drive Summary" }).catch(() => downloadBlob(blob, "weekly-drive-summary.png"));
        } else {
          downloadBlob(blob, "weekly-drive-summary.png");
        }
        el.classList.remove("share-mode"); window.clearTimeout(safety); setSharing(false);
      }, "image/png");
    } catch (e) {
      console.error("Failed to capture weekly summary", e); el.classList.remove("share-mode"); setSharing(false);
    }
  }, [sharing]);

  return (
    <div className="weekly-overlay">
      <div ref={weeklyRef} className="overlay-card weekly-summary-card">
        <header className="weekly-header">
          <div>
            <h1>Weekly Detail Summary</h1>
            {rangeLabel ? <p className="weekly-range">{rangeLabel}</p> : null}
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button className="summary-icon-btn" type="button" aria-label="Share Report" title="Share Report" onClick={() => handleWeeklyShare()}>
              {/* Share / Export Icon (inverted style) */}
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 15V3" />
                <path d="M7 10l5-5 5 5" />
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              </svg>
            </button>
            <button className="summary-icon-btn summary-icon-danger" type="button" onClick={onClose} aria-label="Close" title="Close">
              {/* Reuse cross icon */}
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </button>
          </div>
        </header>
        {loading && <p className="weekly-loading">Loading weekly report...</p>}
        {error && <p className="weekly-error" role="alert">{error}</p>}
        {summary && summary.session_count > 0 ? (
          <>
            <div className="weekly-stats">
              <div><span className="weekly-stat-label">Sessions</span><span className="weekly-stat-value">{summary.session_count}</span></div>
              <div><span className="weekly-stat-label">Drive Time</span><span className="weekly-stat-value">{totalDriveTime}</span></div>
              <div><span className="weekly-stat-label">Normal</span><span className="weekly-stat-value">{normalPercent}%</span></div>
              <div><span className="weekly-stat-label">Arrhythmia</span><span className="weekly-stat-value">{arrhythmiaPercent}%</span></div>
              <div><span className="weekly-stat-label">Avg Heart Rate</span><span className="weekly-stat-value">{avgHeartRate !== null ? `${avgHeartRate} bpm` : "—"}</span></div>
              <div><span className="weekly-stat-label">Avg IBI</span><span className="weekly-stat-value">{avgIbi !== null ? `${avgIbi} ms` : "—"}</span></div>
            </div>
            <div className="weekly-session-list">
              <h2>Drive Log</h2>
              <ul>
                {sessions.slice(0, 4).map((session) => {
                  const endedAt = new Date(session.ended_at);
                  const endedLabel = Number.isNaN(endedAt.getTime()) ? session.ended_at : endedAt.toLocaleString();
                  const duration = formatDurationSeconds(session.duration_seconds);
                  const score = Number.isFinite(session.score) ? Math.round(session.score) : null;
                  const heartRate = Number.isFinite(session.mean_heart_rate) ? `${session.mean_heart_rate.toFixed(1)} bpm` : "—";
                  const signal = Number.isFinite(session.mean_signal_quality) ? `${Math.round(session.mean_signal_quality * 100)}%` : "—";
                  const confidence = Number.isFinite(session.mean_confidence) ? `${Math.round(session.mean_confidence * 100)}%` : "—";
                  const ibi = Number.isFinite(session.mean_ibi_ms) ? `${session.mean_ibi_ms.toFixed(0)} ms` : "—";
                  return (
                    <li key={session.session_id} className="weekly-session-item">
                      <div className="weekly-session-meta">
                        <span className="weekly-session-time">{endedLabel}</span>
                        <span className="weekly-session-duration">{duration}</span>
                      </div>
                      <div className="weekly-session-metrics">
                        <span>Score: {score ?? "—"}</span>
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
        ) : (!loading && !error) ? <p className="weekly-empty">No drives recorded in the last seven days.</p> : null}
      </div>
    </div>
  );
}

function formatWeeklyRange(startIso: string, endIso: string): string {
  const start = new Date(startIso); const end = new Date(endIso);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return "";
  const sameYear = start.getFullYear() === end.getFullYear();
  const dateFmt = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" });
  const yearFmt = new Intl.DateTimeFormat(undefined, { year: "numeric" });
  const sLabel = dateFmt.format(start); const eLabel = dateFmt.format(end);
  const sYear = yearFmt.format(start); const eYear = yearFmt.format(end);
  return sameYear ? `${sLabel} – ${eLabel}, ${eYear}` : `${sLabel}, ${sYear} – ${eLabel}, ${eYear}`;
}

function formatDurationSeconds(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds <= 0) return "0m";
  const whole = Math.round(totalSeconds); const h = Math.floor(whole / 3600); const m = Math.floor((whole % 3600) / 60); const s = whole % 60; const parts: string[] = []; if (h) parts.push(`${h}h`); if (m) parts.push(`${m}m`); if (!parts.length && s) parts.push(`${s}s`); return parts.join(" ") || "0m";
}

function formatDuration(ms: number) {
  if (!ms || ms < 1000) return "under 1 minute";
  const secs = Math.round(ms / 1000); const m = Math.floor(secs / 60); const s = secs % 60; const minPart = m ? `${m} min${m === 1 ? "" : "s"}` : ""; const secPart = s ? `${s} sec${s === 1 ? "" : "s"}` : ""; return [minPart, secPart].filter(Boolean).join(" ") || "under 1 minute";
}

function prettifyLabel(label?: string): string {
  if (!label) return "Unknown";
  return label.toLowerCase().split(/[_\s]+/).filter(Boolean).map(p => p[0].toUpperCase() + p.slice(1)).join(" ");
}

/**
 * Lightweight arrhythmia label detector used in summary aggregations.
 */
function isArrhythmiaLabel(raw: string): boolean {
  if (!raw) return false;
  const s = raw.toLowerCase();
  const keywords = ["arrhythm", "arrhyth", "fibril", "ventricular", "vfib", "vtach", "asyst", "afib", "atrial", "flutter"];
  return keywords.some((k) => s.includes(k));
}



function deriveSeverity(label: string, status: string): "normal" | "caution" | "critical" {
  const l = (label || "").toLowerCase();
  const st = (status || "").toLowerCase();
  // Map clearly severe rhythms to critical
  if (
    l.includes("asyst") ||
    l.includes("ventricular") ||
    l.includes("fibrillation") ||
    l.includes("arrhythm") ||
    l.includes("arrhyth") ||
    st === "emergency"
  )
    return "critical";

  // Elevated conditions
  if (l.includes("tach") || l.includes("brady") || st === "caution") return "caution";

  // Default to normal for healthy/neutral labels
  return "normal";
}

function describeSeverity(sev: "normal" | "caution" | "critical") {
  return sev === "critical" ? "Live Critical" : sev === "caution" ? "Live Elevated" : "Live Stable";
}

type ScoreTone = "stable" | "warning" | "critical";

function getScoreToneMeta(tone: ScoreTone, score: number) {
  switch (tone) {
    case "stable":
      return { title: "Guarded & Clear", message: "Rhythms stayed within expected thresholds.", caption: "Recovered Drive" };
    case "warning":
      return { title: "Eyes On Recovery", message: "Minor arrhythmias surfaced in bursts.", caption: "Monitor closely" };
    default:
      return { title: "Critical Escalation", message: `Severe runs peaked at ${score}% of the drive.`, caption: "Escalate to clinician" };
  }
}

