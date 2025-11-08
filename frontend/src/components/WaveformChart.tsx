import { useMemo } from "react";
import type { WaveformSample } from "../types";

interface WaveformChartProps {
  samples: WaveformSample[]; // raw
  maSamples?: WaveformSample[]; // moving average (trailing)
  maLagSamples?: number; // visual left-shift for MA to emphasize lag
}

const WIDTH = 640;
const HEIGHT = 180;

export function WaveformChart({ samples, maSamples, maLagSamples = 12 }: WaveformChartProps) {
  const rawGeom = useMemo(() => buildGeom(samples, WIDTH, HEIGHT, 0, 8), [samples]);
  const maGeom = useMemo(() => buildGeom(maSamples ?? [], WIDTH, HEIGHT, maLagSamples, 8), [maSamples, maLagSamples]);
  return (
    <div className="card">
      <h3 className="section-title">Pulse Waveform</h3>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 6 }}>
        <LegendSwatch color="#22c55e" label="Raw" />
        <LegendSwatch color="#60a5fa" label="MA" />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#64748b", marginBottom: 4 }}>
        <span>Past</span>
        <span>Now →</span>
      </div>
      <svg className="waveform" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none">
        <rect x="0" y="0" width={WIDTH} height={HEIGHT} fill="#0f172a" opacity={0.04} />
        {/* raw waveform (draw first) */}
        <path d={rawGeom.d} stroke="#22c55e" strokeWidth={2} fill="none" opacity={0.9} />
        {/* moving average overlay */}
        {maSamples && maSamples.length > 1 && (
          <path d={maGeom.d} stroke="#60a5fa" strokeWidth={2} fill="none" opacity={0.9} />
        )}
        {/* end-point dots to guide trend */}
        {rawGeom.last && (
          <circle cx={rawGeom.last.x} cy={rawGeom.last.y} r={3} fill="#22c55e" stroke="#0f172a22" strokeWidth={1} />
        )}
        {maSamples && maSamples.length > 1 && maGeom.last && (
          <circle cx={maGeom.last.x} cy={maGeom.last.y} r={3} fill="#60a5fa" stroke="#0f172a22" strokeWidth={1} />
        )}
        {/* right-side now marker */}
        <line x1={WIDTH - 1} y1={0} x2={WIDTH - 1} y2={HEIGHT} stroke="#cbd5e1" strokeDasharray="4 4" strokeWidth={1} />
      </svg>
    </div>
  );
}

function buildGeom(
  samples: WaveformSample[],
  width: number,
  height: number,
  offsetSamples = 0,
  marginRightPx = 8
): { d: string; last?: { x: number; y: number } } {
  if (!samples.length) {
    const mid = height / 2;
    return { d: `M0 ${mid} L${width - marginRightPx} ${mid}` };
  }
  const values = samples.map((s) => s.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const n = samples.length;
  const drawWidth = Math.max(1, width - marginRightPx);
  const step = n > 1 ? drawWidth / (n - 1) : drawWidth;
  let d = "";
  let last: { x: number; y: number } | undefined;
  for (let i = 0; i < n; i++) {
    const x = Math.min(drawWidth, Math.max(0, i - offsetSamples) * step);
    const normalized = (values[i] - min) / range;
    const y = height - normalized * height;
    d += `${i === 0 ? "M" : "L"}${x} ${y}`;
    if (x <= drawWidth) {
      last = { x, y };
    }
    if (i < n - 1) d += " ";
  }
  return { d, last };
}

function LegendSwatch({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "#334155" }}>
      <span style={{ width: 14, height: 4, background: color, borderRadius: 2 }} />
      {label}
    </span>
  );
}
