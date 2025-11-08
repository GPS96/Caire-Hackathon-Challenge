import { useEffect, useRef } from "react";
import { useDashboardStore } from "../store/dashboard";

declare global {
  interface Window {
    faceapi?: any;
  }
}

type UseEmotionOptions = {
  intervalMs?: number;
  modelBaseURL?: string; // where to load model weights from; default tries /models then CDN fallback
};

// Lightweight emotion detection using face-api.js via CDN runtime loading.
// Requires TinyFaceDetector + FaceExpression model weights to be available.
export function useEmotion(videoRef: React.RefObject<HTMLVideoElement>, active: boolean, opts: UseEmotionOptions = {}) {
  const recordEmotion = useDashboardStore((s) => s.recordEmotion);
  const loadedRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const loadingRef = useRef(false);
  const options = { intervalMs: 800, modelBaseURL: "/models", ...opts };

  useEffect(() => {
    const loadScript = async (src: string) => {
      if (document.querySelector(`script[src=\"${src}\"]`)) return;
      await new Promise<void>((resolve, reject) => {
        const s = document.createElement("script");
        s.src = src;
        s.async = true;
        s.onload = () => resolve();
        s.onerror = () => reject(new Error(`Failed to load ${src}`));
        document.head.appendChild(s);
      });
    };

    const tryLoadModels = async () => {
      const faceapi = window.faceapi;
      if (!faceapi) return false;
      try {
        // First try local /models; fall back to CDN if unavailable
        const base = options.modelBaseURL || "/models";
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(base),
          faceapi.nets.faceExpressionNet.loadFromUri(base),
        ]);
        return true;
      } catch {
        try {
          const cdn = "https://cdn.jsdelivr.net/gh/vladmandic/face-api/model";
          await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(cdn),
            faceapi.nets.faceExpressionNet.loadFromUri(cdn),
          ]);
          return true;
        } catch {
          return false;
        }
      }
    };

    const init = async () => {
      if (loadedRef.current || loadingRef.current) return;
      loadingRef.current = true;
      try {
        if (!window.faceapi) {
          await loadScript("https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.min.js");
        }
        const ok = await tryLoadModels();
        loadedRef.current = !!ok;
      } finally {
        loadingRef.current = false;
      }
    };

    const start = async () => {
      await init();
      if (!loadedRef.current) return;
      if (timerRef.current != null) return;
      const faceapi = window.faceapi!;
      const detect = async () => {
        const v = videoRef.current;
        if (!v || v.readyState < 2) return; // metadata not ready
        try {
          const res = await faceapi
            .detectSingleFace(v, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.2 }))
            .withFaceExpressions();
          if (res && res.expressions) {
            // pick top emotion
            let best: { label: string; prob: number } = { label: "unknown", prob: 0 };
            Object.entries(res.expressions).forEach(([label, prob]) => {
              const p = typeof prob === "number" ? prob : 0;
              if (p > best.prob) best = { label, prob: p };
            });
            if (best.label && best.prob > 0.1) {
              recordEmotion(best.label);
            }
          }
        } catch {
          // ignore inference errors
        }
      };
      timerRef.current = window.setInterval(detect, options.intervalMs);
    };

    const stop = () => {
      if (timerRef.current != null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };

    if (active) start();
    else stop();

    return () => stop();
  }, [active, options.intervalMs, options.modelBaseURL, recordEmotion, videoRef]);
}
