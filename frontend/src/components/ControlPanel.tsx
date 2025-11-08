interface ControlPanelProps {
  onCallEMS: () => void;
  onLogEvent: () => void;
}

export function ControlPanel({ onCallEMS, onLogEvent }: ControlPanelProps) {
  return (
    <div className="card">
      <h3 className="section-title">Actions</h3>
      <div className="controls">
        <button className="primary" onClick={onCallEMS} type="button">
          Call EMS
        </button>
        <button className="secondary" onClick={onLogEvent} type="button">
          Log Event
        </button>
      </div>
    </div>
  );
}
