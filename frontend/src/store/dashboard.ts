import { create } from "zustand";
import { ArrhythmiaPushPayload, InferResponsePayload, WaveformSample } from "../types";

export interface SessionLabelBreakdown {
  labelKey: string;
  displayName: string;
  count: number;
  percentage: number;
}

export interface SessionSummary {
  totalSamples: number;
  breakdown: SessionLabelBreakdown[];
  emotionBreakdown?: SessionLabelBreakdown[];
  dominantLabel: string;
  score: number;
  startedAt?: string;
  endedAt: string;
  durationMs?: number;
  sessionId: string;
  meanHeartRate: number;
  meanSignalQuality: number;
  meanConfidence: number;
  meanIbiMs: number;
  meanFatigue?: number | null;
}

export interface DashboardState {
  heartRate: number;
  ibi: number[];
  arrhythmiaState: string;
  confidence: number;
  signalQuality: number;
  status: string;
  waveform: WaveformSample[];
  maWaveform: WaveformSample[];
  lastUpdated?: string;
  monitoringActive: boolean;
  setFromInference: (payload: InferResponsePayload) => void;
  syncFromPush: (payload: ArrhythmiaPushPayload) => void;

  // Emergency navigation state
  severeArrhythmiaCount: number;
  navigationVisible: boolean;
  navigationLoading: boolean;
  navigationDestination?: { name: string; address: string };
  navigationSilenced: boolean;
  setNavigationVisible: (visible: boolean) => void;
  setNavigationLoading: (loading: boolean) => void;
  setNavigationDestination: (dest?: { name: string; address: string }) => void;
  resetSevereArrhythmiaCount: () => void;
  silenceNavigation: () => void;
  clearNavigationSilence: () => void;
  navigationPromptVisible: boolean;
  setNavigationPromptVisible: (visible: boolean) => void;
  setMonitoringActive: (active: boolean) => void;
  sessionLabelCounts: Record<string, number>;
  sessionEmotionCounts: Record<string, number>;
  sessionSampleCount: number;
  sessionEmotionSampleCount: number;
  sessionStartedAt?: string;
  sessionSummary?: SessionSummary;
  sessionMetricTotals: SessionMetricTotals;
  currentSessionId?: string;
  resetSessionAnalytics: () => void;
  finalizeSessionSummary: () => void;
  recordEmotion: (emotion: string) => void;
}

const MAX_WAVEFORM_POINTS = 512;
const EMERGENCY_ALERT_THRESHOLD = 5;

export type SessionMetricTotals = {
  heartRateSum: number;
  signalQualitySum: number;
  confidenceSum: number;
  ibiSum: number;
  ibiCount: number;
  sampleCount: number;
  fatigueSum?: number;
  fatigueCount?: number;
};

const createEmptyMetricTotals = (): SessionMetricTotals => ({
  heartRateSum: 0,
  signalQualitySum: 0,
  confidenceSum: 0,
  ibiSum: 0,
  ibiCount: 0,
  sampleCount: 0,
  fatigueSum: 0,
  fatigueCount: 0,
});

const generateSessionId = (): string => {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `session-${Math.random().toString(36).slice(2, 10)}`;
};

type SetState = (
  partial:
    | DashboardState
    | Partial<DashboardState>
    | ((state: DashboardState) => DashboardState | Partial<DashboardState>),
  replace?: boolean
) => void;

export const useDashboardStore = create<DashboardState>()((set: SetState) => ({
  heartRate: 0,
  ibi: [],
  arrhythmiaState: "unknown",
  confidence: 0,
  signalQuality: 0,
  status: "Info",
  waveform: [],
  maWaveform: [],
  severeArrhythmiaCount: 0,
  navigationVisible: false,
  navigationLoading: false,
  navigationDestination: undefined,
  navigationSilenced: false,
  navigationPromptVisible: false,
  monitoringActive: false,
  sessionLabelCounts: {},
  sessionEmotionCounts: {},
  sessionSampleCount: 0,
  sessionEmotionSampleCount: 0,
  sessionStartedAt: undefined,
  sessionSummary: undefined,
  sessionMetricTotals: createEmptyMetricTotals(),
  currentSessionId: undefined,
  setNavigationVisible: (visible: boolean) =>
    set((state) => ({
      navigationVisible: visible,
      navigationPromptVisible: visible ? false : state.navigationPromptVisible,
      navigationSilenced: visible ? false : state.navigationSilenced,
    })),
  setNavigationLoading: (loading: boolean) => set(() => ({ navigationLoading: loading })),
  setNavigationDestination: (dest?: { name: string; address: string }) => set(() => ({ navigationDestination: dest })),
  resetSevereArrhythmiaCount: () => set(() => ({ severeArrhythmiaCount: 0, navigationPromptVisible: false })),
  silenceNavigation: () => set(() => ({ navigationSilenced: true, navigationVisible: false, navigationPromptVisible: false })),
  clearNavigationSilence: () => set(() => ({ navigationSilenced: false })),
  setNavigationPromptVisible: (visible: boolean) => set(() => ({ navigationPromptVisible: visible })),
  setMonitoringActive: (active: boolean) =>
    set((state) => {
      if (active) {
        const sessionId = generateSessionId();
        return {
          monitoringActive: true,
          severeArrhythmiaCount: 0,
          navigationVisible: false,
          navigationLoading: false,
          navigationDestination: undefined,
          navigationSilenced: false,
          navigationPromptVisible: false,
          sessionLabelCounts: {},
          sessionEmotionCounts: {},
          sessionSampleCount: 0,
          sessionEmotionSampleCount: 0,
          sessionStartedAt: new Date().toISOString(),
          sessionSummary: undefined,
          sessionMetricTotals: createEmptyMetricTotals(),
          currentSessionId: sessionId,
        };
      }
      return {
        monitoringActive: false,
        navigationVisible: false,
        navigationLoading: false,
        navigationDestination: undefined,
        navigationSilenced: true,
        navigationPromptVisible: false,
        currentSessionId: state.currentSessionId,
        sessionMetricTotals: state.sessionMetricTotals,
      };
    }),
  resetSessionAnalytics: () =>
    set(() => ({
      sessionLabelCounts: {},
      sessionEmotionCounts: {},
      sessionSampleCount: 0,
      sessionEmotionSampleCount: 0,
      sessionStartedAt: undefined,
      sessionSummary: undefined,
      navigationPromptVisible: false,
      sessionMetricTotals: createEmptyMetricTotals(),
      currentSessionId: undefined,
    })),
  finalizeSessionSummary: () =>
    set((state) => {
      if (state.sessionSampleCount <= 0) {
        return {
          monitoringActive: false,
          navigationVisible: false,
          navigationLoading: false,
          navigationDestination: undefined,
          navigationSilenced: true,
          sessionSummary: undefined,
          sessionStartedAt: undefined,
          sessionMetricTotals: createEmptyMetricTotals(),
          currentSessionId: undefined,
        };
      }

      const total = state.sessionSampleCount;
      const breakdownEntries = Object.entries(state.sessionLabelCounts)
        .map(([labelKey, count]) => ({
          labelKey,
          displayName: prettifyLabel(labelKey),
          count,
          percentage: Math.round((count / total) * 1000) / 10,
        }))
        .sort((a, b) => b.count - a.count);

      // Ensure percentages sum to 100 by adjusting the largest bucket if needed
      const percentageSum = breakdownEntries.reduce((acc, item) => acc + item.percentage, 0);
      if (breakdownEntries.length > 0 && Math.abs(percentageSum - 100) >= 0.1) {
        const delta = Math.round((100 - percentageSum) * 10) / 10;
        breakdownEntries[0].percentage = Math.max(0, Math.min(100, breakdownEntries[0].percentage + delta));
      }
      const normalBucket = breakdownEntries.find((item) => item.labelKey === "normal");
      const score = normalBucket ? Math.round(normalBucket.percentage) : Math.max(0, 100 - Math.round(breakdownEntries[0]?.percentage ?? 0));

      const startedAt = state.sessionStartedAt ? new Date(state.sessionStartedAt).toISOString() : undefined;
      const endedAt = new Date().toISOString();
      const durationMs = startedAt ? Math.max(0, new Date(endedAt).getTime() - new Date(startedAt).getTime()) : undefined;

      const totals = state.sessionMetricTotals;
      const metricSampleCount = totals.sampleCount > 0 ? totals.sampleCount : total;
      const meanHeartRate = metricSampleCount > 0 ? roundToOneDecimal(totals.heartRateSum / metricSampleCount) : 0;
      const meanSignalQuality = metricSampleCount > 0 ? roundToOneDecimal(totals.signalQualitySum / metricSampleCount) : 0;
      const meanConfidence = metricSampleCount > 0 ? roundToOneDecimal(totals.confidenceSum / metricSampleCount) : 0;
      const meanIbiMs = totals.ibiCount > 0 ? roundToOneDecimal(totals.ibiSum / totals.ibiCount) : 0;
  const meanFatigue = (totals.fatigueCount && totals.fatigueCount > 0) ? roundToOneDecimal((totals.fatigueSum || 0) / (totals.fatigueCount || 1)) : null;
      const sessionId = state.currentSessionId ?? generateSessionId();

      // Emotion breakdown
      const emotionTotal = state.sessionEmotionSampleCount;
      const emotionBreakdownEntries = emotionTotal > 0
        ? Object.entries(state.sessionEmotionCounts)
            .map(([labelKey, count]) => ({
              labelKey,
              displayName: prettifyLabel(labelKey),
              count,
              percentage: Math.round((count / emotionTotal) * 1000) / 10,
            }))
            .sort((a, b) => b.count - a.count)
        : [];

      // Adjust emotion percentage sum if needed
      const emoSum = emotionBreakdownEntries.reduce((acc, item) => acc + item.percentage, 0);
      if (emotionBreakdownEntries.length > 0 && Math.abs(emoSum - 100) >= 0.1) {
        const delta = Math.round((100 - emoSum) * 10) / 10;
        emotionBreakdownEntries[0].percentage = Math.max(0, Math.min(100, emotionBreakdownEntries[0].percentage + delta));
      }

      return {
        monitoringActive: false,
        navigationVisible: false,
        navigationLoading: false,
        navigationDestination: undefined,
        navigationSilenced: true,
        sessionSummary: {
          totalSamples: total,
          breakdown: breakdownEntries,
          emotionBreakdown: emotionBreakdownEntries,
          dominantLabel: breakdownEntries[0]?.displayName ?? "Unknown",
          score,
          startedAt,
          endedAt,
          durationMs,
          sessionId,
          meanHeartRate,
          meanSignalQuality,
          meanConfidence,
            meanIbiMs,
            meanFatigue,
        },
        sessionMetricTotals: createEmptyMetricTotals(),
        currentSessionId: undefined,
      };
    }),
  recordEmotion: (emotion: string) =>
    set((state) => {
      if (!state.monitoringActive) return {} as Partial<DashboardState>;
      const key = normalizeLabelKey(emotion);
      const counts = { ...state.sessionEmotionCounts };
      counts[key] = (counts[key] || 0) + 1;
      return {
        sessionEmotionCounts: counts,
        sessionEmotionSampleCount: state.sessionEmotionSampleCount + 1,
      };
    }),
  setFromInference: (payload: InferResponsePayload) =>
    set((state) => {
      const waveform = payload.waveform.slice(-MAX_WAVEFORM_POINTS);

      let sessionLabelCounts = state.sessionLabelCounts;
      let sessionSampleCount = state.sessionSampleCount;
      let sessionMetricTotals = state.sessionMetricTotals;

      if (!state.monitoringActive) {
        return {
          heartRate: payload.heart_rate_bpm,
          ibi: payload.ibi_ms,
          arrhythmiaState: payload.arrhythmia_state,
          confidence: payload.confidence,
          signalQuality: payload.signal_quality,
          status: payload.status,
          waveform,
          maWaveform: movingAverageWaveform(waveform),
          lastUpdated: new Date().toISOString(),
          navigationVisible: false,
          navigationLoading: false,
          navigationDestination: undefined,
          severeArrhythmiaCount: 0,
          navigationSilenced: true,
          navigationPromptVisible: false,
          sessionLabelCounts,
          sessionSampleCount,
          sessionMetricTotals,
        };
      }

      const label = payload.arrhythmia_state || "";
      const labelLower = label.toLowerCase();
      const prevLabelLower = (state.arrhythmiaState || "").toLowerCase();
      const labelChanged = labelLower !== prevLabelLower;
      const isSevere = isArrhythmiaLabel(labelLower);

      let severeArrhythmiaCount = isSevere ? state.severeArrhythmiaCount + 1 : 0;
      let navigationSilenced = isSevere ? state.navigationSilenced : false;
      let navigationVisible = state.navigationVisible;
      let navigationPromptVisible = state.navigationPromptVisible;

      if (!isSevere) {
        navigationVisible = false;
        navigationSilenced = false;
        navigationPromptVisible = false;
      } else {
        if (navigationSilenced && labelChanged) {
          navigationSilenced = false;
          severeArrhythmiaCount = 1;
        }
        if (!navigationSilenced) {
          if (severeArrhythmiaCount >= EMERGENCY_ALERT_THRESHOLD && !state.navigationVisible) {
            navigationPromptVisible = true;
            navigationVisible = false;
          }
        }
      }

      if (state.monitoringActive) {
        const labelKey = normalizeLabelKey(labelLower);
        sessionLabelCounts = incrementLabelCount(sessionLabelCounts, labelKey);
        sessionSampleCount = sessionSampleCount + 1;
        const ibiValues = Array.isArray(payload.ibi_ms) ? payload.ibi_ms : [];
        sessionMetricTotals = accumulateMetricTotals(sessionMetricTotals, {
          heartRate: payload.heart_rate_bpm,
          signalQuality: payload.signal_quality,
          confidence: payload.confidence,
          fatigue: (payload as any).fatigue_score ?? null,
          ibiValues,
        });
      }

      return {
        heartRate: payload.heart_rate_bpm,
        ibi: payload.ibi_ms,
        arrhythmiaState: payload.arrhythmia_state,
        confidence: payload.confidence,
        signalQuality: payload.signal_quality,
        status: payload.status,
        waveform,
        maWaveform: movingAverageWaveform(waveform),
        lastUpdated: new Date().toISOString(),
        navigationVisible,
        navigationLoading: navigationVisible ? state.navigationLoading : false,
        navigationDestination: navigationVisible ? state.navigationDestination : undefined,
        severeArrhythmiaCount,
        navigationSilenced,
        navigationPromptVisible,
        sessionLabelCounts,
        sessionSampleCount,
        sessionMetricTotals,
      };
    }),
  // Prefer backend-provided enriched fields from websocket; if absent, synthesize minimal values
  // so the dashboard remains informative until real-time /infer responses are wired in.
  syncFromPush: (payload: ArrhythmiaPushPayload) =>
    set((state) => {
      const hr = payload.heart_rate_bpm || state.heartRate || 60;
      const now = new Date();
      const t = now.getTime() / 1000; // seconds
      const freqHz = Math.max(0.6, Math.min(3.0, hr / 60)); // 36–180 bpm range
      const syntheticValue = Math.sin(2 * Math.PI * freqHz * t) * 0.6 + Math.sin(2 * Math.PI * freqHz * 2 * t) * 0.2;
      const syntheticSample: WaveformSample = { timestamp: now.toISOString(), value: syntheticValue };

      // Approximate IBI from HR (ms between beats)
      const ibiMsFromHr = hr > 0 ? Math.round(60000 / hr) : 0;

      // Smoothly vary signal quality between 0.7–0.95 for demo visuals
      const sqBase = state.signalQuality > 0 ? state.signalQuality : 0.85;
      const syntheticSignalQuality = Math.max(0.7, Math.min(0.95, sqBase + (Math.random() - 0.5) * 0.02));

      const incomingWaveform = payload.waveform && payload.waveform.length > 0 ? payload.waveform : undefined;
      const waveform = incomingWaveform
        ? [...state.waveform, ...incomingWaveform].slice(-MAX_WAVEFORM_POINTS)
        : [...state.waveform, syntheticSample].slice(-MAX_WAVEFORM_POINTS);
      const maWaveform = movingAverageWaveform(waveform);
      const maHigh = isMAHigh(maWaveform);

      // Smooth top-level arrhythmia label: require a short run-length before switching
      const nextLabel = payload.arrhythmia_state;
      const prevLabel = state.arrhythmiaState || "unknown";
      const labelStable = prevLabel === nextLabel;
      let displayLabel = nextLabel;
      if (!labelStable) {
        const prevKnown = prevLabel.toLowerCase() !== "unknown";
        const allowSwitch = payload.confidence >= 0.6 || maHigh || !prevKnown;
        if (!allowSwitch) {
          displayLabel = prevLabel;
        }
      }
      const displayConfidence =
        displayLabel === nextLabel
          ? payload.confidence
          : state.confidence * 0.7 + payload.confidence * 0.3;

      let sessionLabelCounts = state.sessionLabelCounts;
      let sessionSampleCount = state.sessionSampleCount;
      let sessionMetricTotals = state.sessionMetricTotals;

      if (!state.monitoringActive) {
        return {
          heartRate: hr,
          ibi: payload.ibi_ms && payload.ibi_ms.length > 0 ? payload.ibi_ms : ibiMsFromHr ? [ibiMsFromHr] : state.ibi,
          arrhythmiaState: displayLabel,
          confidence: displayConfidence,
          signalQuality: typeof payload.signal_quality === "number" ? payload.signal_quality : syntheticSignalQuality,
          status: payload.status,
          waveform,
          maWaveform,
          lastUpdated: payload.generated_at,
          severeArrhythmiaCount: 0,
          navigationVisible: false,
          navigationSilenced: true,
          navigationLoading: false,
          navigationDestination: undefined,
          navigationPromptVisible: false,
          sessionLabelCounts,
          sessionSampleCount,
          sessionMetricTotals,
        };
      }

      // Emergency navigation trigger logic
  let severeArrhythmiaCount = state.severeArrhythmiaCount;
      const labelLower = displayLabel.toLowerCase();
      const prevLabelLower = (state.arrhythmiaState || "").toLowerCase();
      const labelChanged = labelLower !== prevLabelLower;

      let navigationSilenced = state.navigationSilenced;
      let navigationPromptVisible = state.navigationPromptVisible;
      if (isArrhythmiaLabel(labelLower)) {
        severeArrhythmiaCount = severeArrhythmiaCount + 1;
        if (navigationSilenced && labelChanged) {
          navigationSilenced = false;
          severeArrhythmiaCount = 1;
        }
      } else {
        severeArrhythmiaCount = 0;
        navigationSilenced = false;
        navigationPromptVisible = false;
      }
      let navigationVisible = state.navigationVisible;
      if (!navigationSilenced && severeArrhythmiaCount >= EMERGENCY_ALERT_THRESHOLD && !state.navigationVisible) {
        navigationPromptVisible = true;
        navigationVisible = false;
      }

      const labelKey = normalizeLabelKey(labelLower);
      sessionLabelCounts = incrementLabelCount(sessionLabelCounts, labelKey);
      sessionSampleCount = sessionSampleCount + 1;
      sessionMetricTotals = accumulateMetricTotals(sessionMetricTotals, {
        heartRate: hr,
        signalQuality: typeof payload.signal_quality === "number" ? payload.signal_quality : syntheticSignalQuality,
        confidence: displayConfidence,
        fatigue: (payload as any).fatigue_score ?? null,
        ibiValues: payload.ibi_ms,
        ibiFallback: ibiMsFromHr,
      });

      return {
        heartRate: hr,
        ibi: payload.ibi_ms && payload.ibi_ms.length > 0 ? payload.ibi_ms : ibiMsFromHr ? [ibiMsFromHr] : state.ibi,
        arrhythmiaState: displayLabel,
        confidence: displayConfidence,
        signalQuality: typeof payload.signal_quality === "number" ? payload.signal_quality : syntheticSignalQuality,
        status: payload.status,
        waveform,
        maWaveform,
        lastUpdated: payload.generated_at,
        severeArrhythmiaCount,
        navigationVisible,
        navigationSilenced,
        navigationPromptVisible,
        sessionLabelCounts,
        sessionSampleCount,
        sessionMetricTotals,
      };
    }),
}));

// --- helpers ---
const MA_WINDOW = 25; // larger MA window for clearer trend separation

function normalizeLabelKey(raw: string): string {
  const base = (raw || "unknown").trim().toLowerCase();
  return base.length > 0 ? base : "unknown";
}

function prettifyLabel(label?: string): string {
  if (!label) return "Unknown";
  return label
    .split(/[\s_]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function incrementLabelCount(counts: Record<string, number>, labelKey: string): Record<string, number> {
  const next = { ...counts };
  next[labelKey] = (next[labelKey] || 0) + 1;
  return next;
}

function accumulateMetricTotals(
  totals: SessionMetricTotals,
  sample: {
    heartRate?: number;
    signalQuality?: number;
    confidence?: number;
    fatigue?: number | null;
    ibiValues?: number[];
    ibiFallback?: number | null;
  }
): SessionMetricTotals {
  const next: SessionMetricTotals = { ...totals };
  next.sampleCount = next.sampleCount + 1;

  const hr = toFiniteNumber(sample.heartRate);
  if (hr !== undefined) {
    next.heartRateSum += hr;
  }

  const sq = toFiniteNumber(sample.signalQuality);
  if (sq !== undefined) {
    next.signalQualitySum += sq;
  }

  const confidence = toFiniteNumber(sample.confidence);
  if (confidence !== undefined) {
    next.confidenceSum += confidence;
  }

  const ibiValues = Array.isArray(sample.ibiValues)
    ? sample.ibiValues.filter((value): value is number => typeof value === "number" && Number.isFinite(value))
    : [];
  if (ibiValues.length > 0) {
    next.ibiSum += ibiValues.reduce((acc, value) => acc + value, 0);
    next.ibiCount += ibiValues.length;
  } else {
    const fallback = toFiniteNumber(sample.ibiFallback);
    if (fallback !== undefined) {
      next.ibiSum += fallback;
      next.ibiCount += 1;
    }
  }

  const f = typeof sample.fatigue === "number" && Number.isFinite(sample.fatigue) ? sample.fatigue : undefined;
  if (typeof next.fatigueSum !== "number") next.fatigueSum = 0;
  if (typeof next.fatigueCount !== "number") next.fatigueCount = 0;
  if (f !== undefined) {
    next.fatigueSum = (next.fatigueSum || 0) + f;
    next.fatigueCount = (next.fatigueCount || 0) + 1;
  }

  return next;
}

function toFiniteNumber(value?: number | null): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function roundToOneDecimal(value: number): number {
  return Math.round(value * 10) / 10;
}

function movingAverageWaveform(samples: WaveformSample[]): WaveformSample[] {
  // Trailing (causal) moving average so it acts like a lagging "shadow" of the raw signal
  if (samples.length <= 2) return samples;
  const w = Math.max(1, Math.min(MA_WINDOW, samples.length));
  const values = samples.map((s) => s.value);
  const prefix: number[] = new Array(values.length + 1).fill(0);
  for (let i = 0; i < values.length; i++) prefix[i + 1] = prefix[i] + values[i];
  const out: WaveformSample[] = new Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - (w - 1));
    const end = i;
    const len = end - start + 1;
    const sum = prefix[end + 1] - prefix[start];
    const val = sum / len;
    out[i] = { timestamp: samples[i].timestamp, value: val };
  }
  return out;
}

function isMAHigh(ma: WaveformSample[]): boolean {
  const n = ma.length;
  if (n < 10) return false;
  const tail = ma.slice(-Math.min(40, n)).map((s) => s.value);
  const minV = Math.min(...tail);
  const maxV = Math.max(...tail);
  const range = maxV - minV || 1;
  const last = tail[tail.length - 1];
  const norm = (last - minV) / range;
  // consider "high" if last value is in the top quartile of the recent MA range
  return norm >= 0.75;
}

/**
 * Determine whether a normalized lowercase label string indicates an arrhythmia.
 * We match common substrings (arrhythm, fibrillation, ventricular, asyst, afib, etc.).
 */
function isArrhythmiaLabel(raw: string): boolean {
  if (!raw) return false;
  const s = raw.toLowerCase();
  const keywords = [
    "arrhythm",
    "arrhyth",
    "fibril",
    "fibrillation",
    "ventricular",
    "vfib",
    "vtach",
    "asyst",
    "afib",
    "atrial",
    "flutter",
  ];
  return keywords.some((k) => s.includes(k));
}
