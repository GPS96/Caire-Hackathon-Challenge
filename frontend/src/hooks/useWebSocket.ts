import { useEffect, useRef } from "react";
import { useDashboardStore } from "../store/dashboard";
import type { ArrhythmiaPushPayload } from "../types";
import type { DashboardState } from "../store/dashboard";

// Helper: capture frame from videoRef, encode as base64
export function captureFrameBase64(videoRef: React.RefObject<HTMLVideoElement>): string | null {
  const video = videoRef.current;
  if (!video) return null;
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    return null; // metadata not ready yet
  }
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  // Slightly lower quality to reduce payload size while retaining fidelity
  return canvas.toDataURL("image/jpeg", 0.85).split(",")[1] || null;
}

// Ensure we use the correct WebSocket scheme derived from the backend URL
const BACKEND_BASE = __BACKEND_URL__.replace(/\/$/, "");
const SOCKET_ENDPOINT = `${BACKEND_BASE.replace(/^http/, "ws")}/ws/arrhythmia`;

export function useArrhythmiaSocket(videoRef?: React.RefObject<HTMLVideoElement>, sessionActive?: boolean): void {
  const syncFromPush = useDashboardStore((state: DashboardState) => state.syncFromPush);
  const socketRef = useRef<WebSocket | null>(null);
  const frameTimerRef = useRef<number | null>(null);
  // Match live_pipeline cadence (~30 fps) so upstream API accumulates samples quickly
  const SEND_INTERVAL = 33; // ~30 fps
  
  useEffect(() => {
    const socket = new WebSocket(SOCKET_ENDPOINT);
    socketRef.current = socket;

    const startSendingFrames = () => {
      if (!videoRef || !sessionActive) return;
      const isOpen = socket.readyState === WebSocket.OPEN;
      if (!isOpen || frameTimerRef.current) return;
      console.log("[Frontend] Starting to send frames");
      let frameCount = 0;
      frameTimerRef.current = window.setInterval(() => {
        const frameBase64 = captureFrameBase64(videoRef);
        if (!frameBase64) {
          return;
        }
        frameCount++;
        socket.send(JSON.stringify({ frame_base64: frameBase64, timestamp: new Date().toISOString() }));
        if (frameCount % 60 === 0) {
          console.log(`[Frontend] Sent ${frameCount} frames to backend`);
        }
      }, SEND_INTERVAL);
    };

    const stopSendingFrames = () => {
      if (frameTimerRef.current) {
        window.clearInterval(frameTimerRef.current);
        frameTimerRef.current = null;
        console.log("[Frontend] Stopped sending frames");
      }
    };

    socket.onopen = () => {
      console.log("[Frontend] WebSocket connected to backend");
      startSendingFrames();
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as ArrhythmiaPushPayload;
        syncFromPush(payload);
        console.log("[Frontend] Received payload from backend:", payload);
      } catch (error) {
        console.error("Failed to parse socket payload", error);
      }
    };

    socket.onerror = (error) => {
      console.error("WebSocket error", error);
    };

    socket.onclose = () => {
      console.log("[Frontend] WebSocket closed");
      stopSendingFrames();
    };

    return () => {
      socket.close();
      socketRef.current = null;
      stopSendingFrames();
    };
  }, [syncFromPush, videoRef, sessionActive]);
  
  // Separate effect to manage frame sending based on sessionActive
  useEffect(() => {
    const socket = socketRef.current;
    if (!socket) return;
    if (!sessionActive || socket.readyState !== WebSocket.OPEN) {
      if (frameTimerRef.current) {
        window.clearInterval(frameTimerRef.current);
        frameTimerRef.current = null;
        console.log("[Frontend] Session inactive or socket not open; stopped frames");
      }
      return;
    }

    if (!frameTimerRef.current) {
      console.log("[Frontend] Session active and socket open; ensuring frames are sent");
      frameTimerRef.current = window.setInterval(() => {
        const frameBase64 = captureFrameBase64(videoRef!);
        if (!frameBase64) {
          return;
        }
        socket.send(JSON.stringify({ frame_base64: frameBase64, timestamp: new Date().toISOString() }));
      }, SEND_INTERVAL);
    }

    return () => {
      if (frameTimerRef.current) {
        window.clearInterval(frameTimerRef.current);
        frameTimerRef.current = null;
        console.log("[Frontend] Cleaned up frame timer");
      }
    };
  }, [videoRef, sessionActive]);
}
