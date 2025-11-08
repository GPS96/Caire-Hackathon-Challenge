interface MetricCardProps {
  title: string;
  value: string;
  subtitle?: string;
}

export function MetricCard({ title, value, subtitle }: MetricCardProps) {
  return (
    <div className="card">
      <h3 className="section-title">{title}</h3>
      <p style={{ fontSize: "2.5rem", margin: 0, fontWeight: 700 }}>{value}</p>
      {subtitle ? <p style={{ margin: 0, opacity: 0.65 }}>{subtitle}</p> : null}
    </div>
  );
}
