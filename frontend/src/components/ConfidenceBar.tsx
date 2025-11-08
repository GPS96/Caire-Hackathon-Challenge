interface ConfidenceBarProps {
  confidence: number;
}

export function ConfidenceBar({ confidence }: ConfidenceBarProps) {
  const percent = Math.round(Math.min(Math.max(confidence, 0), 1) * 100);
  return (
    <div className="card">
      <h3 className="section-title">Confidence</h3>
      <div style={{ background: "#e2e8f0", borderRadius: 12, overflow: "hidden", height: 24 }}>
        <div
          style={{
            width: `${percent}%`,
            background: percent > 80 ? "#22c55e" : percent > 60 ? "#eab308" : "#f97316",
            height: "100%",
            transition: "width 0.3s ease",
          }}
        />
      </div>
      <p style={{ margin: "0.5rem 0 0", fontWeight: 600 }}>{percent}%</p>
    </div>
  );
}
