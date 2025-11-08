interface StatusBannerProps {
  status: string;
  arrhythmiaState: string;
  lastUpdated?: string;
}

const STYLE_MAP: Record<string, string> = {
  Info: "status-info",
  Caution: "status-caution",
  Emergency: "status-emergency",
};

const LABEL_CLASS: Record<string, string> = {
  normal: "arr-normal",
  tachycardia: "arr-tachy",
  bradycardia: "arr-brady",
  asystole: "arr-asys",
  ventricular_flutter_fib: "arr-vfib",
};

export function StatusBanner({ status, arrhythmiaState, lastUpdated }: StatusBannerProps) {
  const key = arrhythmiaState?.toLowerCase().replace(/\s+/g, "_") || "normal";
  const labelClass = LABEL_CLASS[key] ?? LABEL_CLASS.normal;
  const classes = `status-banner ${STYLE_MAP[status] ?? STYLE_MAP.Info} ${labelClass}`;
  return (
    <section className={classes}>
      <div>
        <h2 style={{ margin: 0, fontSize: "1.5rem" }}>{arrhythmiaState.toUpperCase()}</h2>
        <p style={{ margin: 0, opacity: 0.85 }}>Current arrhythmia classification</p>
      </div>
      <div style={{ textAlign: "right" }}>
        <strong>{status}</strong>
        <p style={{ margin: 0, opacity: 0.7 }}>Updated: {lastUpdated ?? "—"}</p>
      </div>
    </section>
  );
}
