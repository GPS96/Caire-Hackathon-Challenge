# Caire JS Demo (Browser) — Payload & WebSocket Guide

This is a **single‑file browser client** that streams webcam frames to our backend over **WebSocket** and prints the server’s responses for the heart rate and rppg inferences.

This README focuses on:
- how to **construct the outgoing payload**, and
- how to **open and use the WebSocket**.


---

## 1) Connect to the WebSocket

**URL shape**

```
ws://<HOST>:<PORT>/ws/?api_key=<YOUR_API_KEY>
```

- Only `api_key` is required by the demo. Get a valid key with the caire team.

---

## 2) Outgoing Messages (Client → Server)

The client sends **one message per frame**. The messages should be taged as `"stream"` or `"end"` in the `"state"` field of the message. For the final message, use the `"end"` state. This signals the server that the inference should be over. For the other messages, keep the `"stream"` state.

### Required fields

```jsonc
{
  "datapt_id": "4a4e6d1d-cf3a-48cf-83f8-13a116ffd29e",  // UUIDv4
  "state": "stream",                                   // "stream" or "end"
  "frame_data": "<BASE64_JPEG_NO_PREFIX>",             // JPEG base64, no data URL prefix
  "timestamp": "1730868591.123",                       // UNIX seconds as a string
  "advanced": true                                     // allow advanced streaming data
}
```

- **datapt_id**: a unique identifier for the stream data point. In browser code we generate `crypto.randomUUID()` which is equivalent to Python’s `uuid.uuid4()`.
- **state**: `"stream"` for regular frames, `"end"` for the final message.
- **frame_data**: base64 JPEG **without** `data:image/jpeg;base64,` prefix.
- **timestamp**: UNIX time in seconds (string). Fractional seconds are allowed.
- **advanced**:  
  - `true`: the server will include **advanced continuous signals** (e.g., `advanced.rppg`, `advanced.rppg_timestamps`) in responses.  
  - `false`: the server will not stream those signals.

### Example: “stream” payload (one frame)

```json
{
  "datapt_id": "ed21c799-9edd-4706-9256-0324a7697adb",
  "state": "stream",
  "frame_data": "<base64>",
  "timestamp": "1730868591.123",
  "advanced": true
}
```

### Example: final “end” payload

```json
{
  "datapt_id": "ed21c799-9edd-4706-9256-0324a7697adb",
  "state": "end",
  "frame_data": "<base64_of_last_frame>",
  "timestamp": "1730868621.987",
  "advanced": true
}
```

---

## 3) Incoming Messages (Server → Client)

The server responses are JSON. Two states are expected:

- `state: "ok"` — **intermediate** updates during streaming
- `state: "finished"` — **final** message for the stream

### Example: intermediate response with advanced data

```json
{
  "state": "ok",
  "socket_id": "c73bb619",
  "datapt_id": "0e06d32c-1afb-43de-994a-8293d96d5e05",
  "inference": { "hr": 66 },
  "advanced": {
    "rppg": [0.5120, 0.4296, ...],
    "rppg_timestamps": [1762439955.133, 1762439955.166, ...]
  },
  "confidence": {},
  "feedback": null,
  "model_version": "HR"
}
```

### Example: final response with advanced disabled

```json
{
  "state": "finished",
  "socket_id": "c73bb619",
  "datapt_id": "ed21c799-9edd-4706-9256-0324a7697adb",
  "inference": { "hr": 65 },
  "advanced": null,
  "confidence": {},
  "feedback": null,
  "model_version": "HR"
}
```

> The demo prints the **entire message** as it arrives, without filtering keys.

---

## 4) Running the Demo

1. Open `index.html` in a modern browser (Chrome/Edge/Safari/Firefox).
   - If your browser blocks webcam from `file://`, serve it locally:
     ```bash
     python -m http.server 8080
     # open http://localhost:8080/index.html
     ```
2. Enter your **WS base** (e.g., `ws://localhost:8003/ws/`) and **API key**.
3. Set **FPS**, **number of frames**, and toggle **Advanced** as needed.
4. Click **Start streaming**.  
   The app streams JPEG frames and prints every server message as pretty JSON.

### Notes on performance

- Frames are compressed as JPEG in the browser before sending.
- If you need higher FPS, reduce resolution or JPEG quality in the app’s `toDataURL('image/jpeg', quality)` call.
- The demo uses **text frames** (`JSON.stringify(...)`).

---

## 5) Minimal Code Map

The client is intentionally organized around these blocks:

- **`buildPayload()`** — constructs the outgoing payload for one frame.
- **`connect()`** — opens the WebSocket, sets event handlers.
- **`startStreaming()`** — grabs webcam frames and sends payloads on an interval.
- **`stopOrReset()`** — stops streaming and closes the socket.
- **`logFullMessage()`** — prints the raw server payload as pretty JSON.

Open `index.html` and search for these function names to navigate the code quickly.


---
