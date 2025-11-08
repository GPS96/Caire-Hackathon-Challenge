import { useEffect, useMemo, useRef, useState } from "react";

const favouriteContacts = [
  { name: "112 – Emergency Services", description: "Dispatches local responders" },
  { name: "Primary Cardiologist", description: "Dr. Singh • Mercy Heart Clinic" },
  { name: "Family Contact", description: "Alex Morgan • Spouse" },
];

type EmergencyActionPromptProps = {
  visible: boolean;
  onDismiss: () => void;
  onNavigate: () => void;
  onCall: (contact: string) => void;
};

export function EmergencyActionPrompt({ visible, onDismiss, onNavigate, onCall }: EmergencyActionPromptProps) {
  const [showContacts, setShowContacts] = useState(false);
  const [activeContact, setActiveContact] = useState<string | null>(null);
  const [callPhase, setCallPhase] = useState<"idle" | "dialing" | "ringing">("idle");
  // Keep the alert visible for a minimum time even if parent visibility toggles off briefly
  const MIN_VISIBLE_MS = 8000; // 8 seconds sticky visibility
  const [openedAt, setOpenedAt] = useState<number | null>(null);

  // Simple synthesized ringtone (web audio) when in 'ringing'
  const audioCtxRef = useRef<AudioContext | null>(null);
  const oscRef = useRef<OscillatorNode | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const ringTimerRef = useRef<number | null>(null);
  const ttsInitedRef = useRef(false);

  const speakText = (text: string) => {
    try {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
      // Avoid overlapping too much: cancel previous
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1.0;
      utter.pitch = 1.0;
      utter.volume = 1.0;
      window.speechSynthesis.speak(utter);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    if (visible) {
      // latch open time when it becomes visible
      setOpenedAt(Date.now());
    } else {
      // If parent hides, only reset immediately if sticky window expired
      const withinHold = openedAt !== null && (Date.now() - openedAt) < MIN_VISIBLE_MS;
      if (!withinHold) {
        setShowContacts(false);
        setActiveContact(null);
        setCallPhase("idle");
        stopRingtone();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible]);

  const handleDismiss = () => {
    setOpenedAt(null);
    setShowContacts(false);
    setActiveContact(null);
    setCallPhase("idle");
    stopRingtone();
    onDismiss();
  };

  // Close the prompt locally without notifying parent (used when choosing Navigate)
  // We avoid calling onDismiss() here because the parent onDismiss also calls
  // silenceNavigation(), which would prevent the navigation card from opening.
  const closeLocal = () => {
    setOpenedAt(null);
    setShowContacts(false);
    setActiveContact(null);
    setCallPhase("idle");
    stopRingtone();
  };

  const contactList = useMemo(() => favouriteContacts, []);

  // Transition from dialing to ringing, and start/stop ringtone accordingly
  useEffect(() => {
    let t: number | null = null;
    if (callPhase === "dialing") {
      // After short delay, show ringing and start ringtone
      t = window.setTimeout(() => {
        setCallPhase("ringing");
        startRingtone();
      }, 1200);
    }
    if (callPhase === "idle") {
      stopRingtone();
    }
    return () => {
      if (t != null) window.clearTimeout(t);
    };
  }, [callPhase]);

  const shouldRender = visible || (openedAt !== null && (Date.now() - openedAt) < MIN_VISIBLE_MS);
  if (!shouldRender) return null;

  function startRingtone() {
    try {
      if (audioCtxRef.current) return;
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 440; // A4
      gain.gain.value = 0.0; // start muted
      osc.connect(gain);
      gain.connect(ctx.destination);
      const now = ctx.currentTime;
      osc.start(now);
      audioCtxRef.current = ctx;
      oscRef.current = osc;
      gainRef.current = gain;
      // Pattern: 1s on, 2s off loop
      const tick = () => {
        if (!gainRef.current || !audioCtxRef.current) return;
        const g = gainRef.current;
        g.gain.cancelScheduledValues(audioCtxRef.current.currentTime);
        g.gain.setValueAtTime(0.25, audioCtxRef.current.currentTime);
        setTimeout(() => {
          if (!gainRef.current || !audioCtxRef.current) return;
          g.gain.setValueAtTime(0.0, audioCtxRef.current.currentTime);
        }, 1000);
      };
      tick();
      ringTimerRef.current = window.setInterval(tick, 3000);
    } catch {
      // ignore audio errors (autoplay policies may block)
    }
  }

  function stopRingtone() {
    try {
      if (ringTimerRef.current != null) {
        window.clearInterval(ringTimerRef.current);
        ringTimerRef.current = null;
      }
      if (oscRef.current) {
        try { oscRef.current.stop(); } catch {}
      }
      if (audioCtxRef.current) {
        try { audioCtxRef.current.close(); } catch {}
      }
    } finally {
      oscRef.current = null;
      audioCtxRef.current = null;
      gainRef.current = null;
    }
  }

  return (
    <div className="emergency-overlay" role="dialog" aria-modal="true" aria-labelledby="emergency-response-title">
      <div className="emergency-card">
        <div className="emergency-window-controls">
          <button
            type="button"
            className="summary-icon-btn summary-icon-danger"
            onClick={handleDismiss}
            aria-label="Dismiss"
            title="Dismiss"
          >
            {/* Cross icon */}
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
        <header className="emergency-card-header">
          <span className="emergency-kicker">Guardian Safety Protocol</span>
          <h2 id="emergency-response-title">Repeated arrhythmias detected</h2>
          <p>Choose how you want to respond. You can trigger an emergency call or request guided navigation.</p>
        </header>

        <div className="emergency-ring-grid">
          <button
            type="button"
            className="emergency-ring emergency-ring-sos"
            onClick={() => setShowContacts((prev) => !prev)}
            aria-expanded={showContacts}
          >
            <span className="emergency-ring-inner">
              <span className="emergency-ring-label">Emergency SOS</span>
              <span className="emergency-ring-subtitle">Call from favourites</span>
            </span>
          </button>

          <button
            type="button"
            className="emergency-ring emergency-ring-nav"
            onClick={() => {
              // close prompt immediately (do not silence navigation) then trigger navigation
              closeLocal();
              try { onNavigate(); } catch {};
            }}
          >
            <span className="emergency-ring-inner">
              <span className="emergency-ring-label">Navigate to Care</span>
              <span className="emergency-ring-subtitle">Open smart routing</span>
            </span>
          </button>
        </div>

        {showContacts ? (
          <div className="emergency-contact-panel">
            <h3>Favourite contacts</h3>
            <ul>
              {contactList.map((contact) => (
                <li key={contact.name}>
                  <button
                    type="button"
                    className="emergency-contact-button"
                    onClick={() => {
                      setActiveContact(contact.name);
                      setCallPhase("dialing");
                      speakText(`Connecting to ${contact.name}`);
                      onCall(contact.name);
                    }}
                  >
                    <span className="emergency-contact-name">{contact.name}</span>
                    <span className="emergency-contact-description">{contact.description}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {/* Phone call simulation overlay */}
        {activeContact && callPhase !== "idle" ? (
          <div className="call-overlay" role="dialog" aria-modal="true" aria-label="Calling">
            <div className="call-card">
              <div className="call-avatar" aria-hidden>
                <span>{activeContact.split(" ").map(w => w[0]).slice(0,2).join("")}</span>
              </div>
              <div className="call-name">{activeContact}</div>
              <div className="call-status">
                {callPhase === "dialing" ? "Calling…" : "Ringing…"}
                <span className="call-dots"><i></i><i></i><i></i></span>
              </div>
              <div className="call-actions">
                <button
                  type="button"
                  className="btn btn-alert call-end"
                  onClick={() => {
                    stopRingtone();
                    setActiveContact(null);
                    setCallPhase("idle");
                  }}
                  aria-label="End call"
                >
                  {/* Phone hang-up icon */}
                  <svg viewBox="0 0 24 24" width="24" height="24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path d="M6.62 10.79a15.05 15.05 0 0 1 10.76 0l.76.3c.63.25 1.35.09 1.83-.39l1.03-1.03a1.5 1.5 0 0 0 0-2.12l-1.11-1.11a3 3 0 0 0-3.13-.7c-2.78 1.02-5.98 1.02-8.76 0a3 3 0 0 0-3.13.7L3.66 7.55a1.5 1.5 0 0 0 0 2.12l1.03 1.03c.48.48 1.2.64 1.83.39l.1-.04z" fill="currentColor"/>
                  </svg>
                </button>
              </div>
            </div>
          </div>
        ) : null}

        {/* Dismiss moved to top-right cross icon */}
      </div>
    </div>
  );
}
