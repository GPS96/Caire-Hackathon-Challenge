"""
Navigation agent using LangGraph + LangChain (modern APIs)

Key features:
- No keyword hardcoding. The LLM chooses tools via tool-calling.
- Two tools: find_nearest_place (POI search) and geocode_address (start navigation).
- Uses geopy Nominatim with a RateLimiter to respect OpenStreetMap policies.
- Flask UI with Leaflet map. POST /agent_query accepts free-form text and optional user_lat/user_lon.

Requirements (install in venv):
  pip install Flask langchain-core langchain-openai langgraph geopy pydantic
  # and set OPENAI_API_KEY in your environment.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from dotenv import load_dotenv, find_dotenv
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# LangChain/LangGraph (modern imports)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Optional: Google Gemini support (generous free tier)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Optional: Ollama support for local models (no API costs)
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# -----------------------------
# Geocoding helpers (robust & rate-limited)
# -----------------------------
_geolocator = Nominatim(user_agent="healthai_nav_agent", timeout=15)
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1)

DEFAULT_LAT = 52.5200
DEFAULT_LON = 13.4050
DEFAULT_RADIUS_KM = 5.0


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


@tool("find_nearest_place")
def find_nearest_place(place_type: str, user_lat: float = DEFAULT_LAT, user_lon: float = DEFAULT_LON, radius_km: float = DEFAULT_RADIUS_KM) -> Dict[str, Any]:
    """Find the nearest place of a given type near user_lat,user_lon within radius_km.

    Parameters:
        place_type: e.g. "hospital", "police station", "rest stop", "pharmacy".
        user_lat: latitude in decimal degrees.
        user_lon: longitude in decimal degrees.
        radius_km: search radius in kilometers.

    Returns a dict with {type, place_type, address, distance, lat, lon} or {error}.
    """
    user_lat = _as_float(user_lat, DEFAULT_LAT)
    user_lon = _as_float(user_lon, DEFAULT_LON)
    radius_km = _as_float(radius_km, DEFAULT_RADIUS_KM)

    query = f"{place_type} near {user_lat}, {user_lon}"
    try:
        results = _geocode(query, exactly_one=False, addressdetails=True, limit=15)
    except Exception as e:
        return {"error": f"Geocoding error: {e}"}

    if not results:
        return {"error": f"No nearby {place_type} found."}

    candidates: List[Dict[str, Any]] = []
    for res in results:
        try:
            lat = float(res.latitude)
            lon = float(res.longitude)
            d_km = geodesic((user_lat, user_lon), (lat, lon)).km
            if d_km <= radius_km:
                candidates.append({
                    "address": getattr(res, "address", str(res)),
                    "distance_km": d_km,
                    "lat": lat,
                    "lon": lon,
                })
        except Exception:
            continue

    if not candidates:
        return {"error": f"No {place_type} found within {radius_km} km."}

    candidates.sort(key=lambda x: x["distance_km"])
    nearest = candidates[0]
    return {
        "type": "nearest_place",
        "place_type": place_type,
        "address": nearest["address"],
        "distance": f"{nearest['distance_km']:.2f} km",
        "lat": nearest["lat"],
        "lon": nearest["lon"],
    }


@tool("geocode_address")
def geocode_address(address: str) -> Dict[str, Any]:
    """Geocode a free-form address and return coordinates and normalized address.

    Parameters:
        address: destination address or place name.

    Returns a dict with {type, lat, lon, address} or {error}.
    """
    try:
        loc = _geocode(address)
    except Exception as e:
        return {"error": f"Geocoding error: {e}"}

    if not loc:
        return {"error": "Could not find coordinates for the given address."}
    return {
        "type": "navigation_coords",
        "lat": float(loc.latitude),
        "lon": float(loc.longitude),
        "address": getattr(loc, "address", address),
    }


# -----------------------------
# Build LangGraph agent
# -----------------------------
TOOLS = [find_nearest_place, geocode_address]

SYSTEM_PROMPT = (
    "You are a helpful navigation assistant with access to two tools:\n\n"
    "TOOLS:\n"
    "1. find_nearest_place(place_type, user_lat, user_lon, radius_km) - Find nearby locations\n"
    "2. geocode_address(address) - Get precise coordinates for navigation\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "- When user asks to find a place (hospital, pharmacy, etc.), ALWAYS call find_nearest_place\n"
    "- After showing results, ask if they want navigation\n"
    "- When user confirms (yes/sure/okay/navigate/go there), you MUST call geocode_address with the exact address you just found\n"
    "- IMPORTANT: Remember the address from your previous tool call and use it with geocode_address\n"
    "- For direct navigation requests ('navigate to X'), call geocode_address immediately\n\n"
    "EXAMPLE CONVERSATION:\n"
    "User: 'Find nearest hospital'\n"
    "You: [Call find_nearest_place('hospital', lat, lon)] → Result shows 'City Hospital, 123 Main St, 2.3km'\n"
    "You: 'I found City Hospital at 123 Main St, 2.3 km away. Would you like navigation there?'\n"
    "User: 'Yes'\n"
    "You: [Call geocode_address('City Hospital, 123 Main St')] → Result shows coordinates\n"
    "You: 'Starting navigation to City Hospital!'\n\n"
    "REMEMBER: When user says yes/okay after you show them a place, you MUST call the geocode_address tool with that address!"
)


def build_graph() -> Any:
    # Determine which LLM provider to use based on environment variables
    provider = os.getenv("LLM_PROVIDER", "").lower()  # "openai", "gemini", or "ollama"

    if provider == "gemini" or (not provider and os.getenv("GOOGLE_API_KEY")):
        # Google Gemini (generous free tier: 1500 requests/day)
        if not GEMINI_AVAILABLE:
            raise ValueError("Gemini selected but langchain-google-genai not installed. Run: pip install langchain-google-genai")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment. Get one free at: https://makersuite.google.com/app/apikey")

        # Prefer a broadly available model; allow override via .env GEMINI_MODEL
        # If you see 404 for v1beta, try: gemini-1.5-flash-8b
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-8b")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_retries=2,  # Retry on rate limits
            request_timeout=30
        )
        print(f"✓ Using Google Gemini: {model_name} (Free tier: 1500 requests/day)")

    elif provider == "ollama" or (not provider and os.getenv("USE_OLLAMA") == "true"):
        # Local Ollama (completely free, runs on your machine)
        if not OLLAMA_AVAILABLE:
            raise ValueError("Ollama selected but langchain-ollama not installed. Run: pip install langchain-ollama")

        model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
        llm = ChatOllama(model=model_name, temperature=0)
        print(f"✓ Using local Ollama: {model_name} (100% free, offline)")

    elif os.getenv("OPENAI_API_KEY"):
        # OpenAI (paid, requires credits)
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model_name, temperature=0)
        print(f"✓ Using OpenAI: {model_name}")

    else:
        raise ValueError(
            "No LLM configured. Choose one:\n"
            "1. GEMINI (FREE): Set GOOGLE_API_KEY in .env (get key: https://makersuite.google.com/app/apikey)\n"
            "2. OpenAI (PAID): Set OPENAI_API_KEY in .env\n"
            "3. Ollama (FREE, LOCAL): Install Ollama and set USE_OLLAMA=true\n"
            "\nRecommended: Use Gemini for free API or Ollama for offline."
        )
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Define the agent node as a proper function
    def call_agent(state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent node that calls LLM with system prompt and conversation history"""
        messages = state["messages"]
        
        # Prepend system message if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        else:
            messages_with_system = messages
        
        # Call LLM
        response = llm_with_tools.invoke(messages_with_system)
        
        # Debug: Print if LLM is calling tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔧 Agent calling tools: {[tc['name'] for tc in response.tool_calls]}")
        else:
            print(f"💬 Agent responding with text: {response.content[:100] if response.content else 'empty'}")
        
        # Return updated state
        return {"messages": [response]}
    
    # Build the graph
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    
    # Add conditional routing: if agent called tools, go to tools node; otherwise end
    graph.add_conditional_edges(
        "agent",
        tools_condition,  # This checks if last message has tool_calls
    )
    
    # After tools execute, go back to agent to format final response
    graph.add_edge("tools", "agent")
    
    compiled_graph = graph.compile()

    # Update active LLM signature for auto-rebuild checks
    global GRAPH_SIG
    GRAPH_SIG = _current_llm_signature() if ' _current_llm_signature' in globals() or '_current_llm_signature' in globals() else None
    try:
        print(f"LLM provider selection: {GRAPH_SIG}")
    except Exception:
        pass

    return compiled_graph


GRAPH = None  # Lazy init
GRAPH_SIG = None  # Tracks active provider/model signature

def _current_llm_signature() -> str:
  provider = os.getenv("LLM_PROVIDER", "").lower()
  if provider == "gemini" or (not provider and os.getenv("GOOGLE_API_KEY")):
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return f"gemini::{model}"
  if provider == "ollama" or (not provider and os.getenv("USE_OLLAMA") == "true"):
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    return f"ollama::{model}"
  if os.getenv("OPENAI_API_KEY"):
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return f"openai::{model}"
  return "none"


def run_agent(query: str, user_lat: Optional[float] = None, user_lon: Optional[float] = None, conversation_history: Optional[List[Any]] = None) -> Dict[str, Any]:
  """Run the graph on a user query with conversation history support."""
  # Reload .env to pick up changes and allow overriding during dev
  try:
    detected_env = find_dotenv(filename=".env", raise_error_if_not_found=False, usecwd=True)
    if detected_env:
      load_dotenv(detected_env, override=True)
  except Exception:
    pass

  global GRAPH, GRAPH_SIG
  sig_now = _current_llm_signature()
  if GRAPH is None or GRAPH_SIG != sig_now:
    GRAPH = build_graph()

  # Build the conversation with history
  if conversation_history is None:
    conversation_history = []

  # Optionally steer LLM with coordinates context
  location_hint = ""
  if user_lat is not None and user_lon is not None:
    location_hint = f"\n(User location: {user_lat}, {user_lon})"

  # Add user message to conversation
  messages = conversation_history + [
    HumanMessage(content=(query + location_hint)),
  ]

  result = GRAPH.invoke({"messages": messages})

  # The graph returns a dict-like state with "messages" (list of BaseMessage)
  msgs: List[Any] = result.get("messages", [])

  if not msgs:
    return {"ok": False, "error": "No messages in graph response", "history": msgs}

  # Extract ALL tool results (not just the first one)
  tool_results: List[Any] = []
  ai_response: Optional[AIMessage] = None

  for m in msgs:
    if isinstance(m, ToolMessage):
      try:
        # ToolMessage.content may be a stringified JSON or a dict
        if isinstance(m.content, str):
          parsed = json.loads(m.content)
          tool_results.append(parsed)
        elif isinstance(m.content, dict):
          tool_results.append(m.content)
      except Exception:
        # If parsing fails, keep the raw content
        tool_results.append({"raw": m.content})
    elif isinstance(m, AIMessage):
      # Keep the latest AI response
      ai_response = m

  # Return tool results, AI text response, plus conversation history
  output: Dict[str, Any] = {}

  # Include ALL tool results (for multi-step workflows)
  if tool_results:
    output["tool_results"] = tool_results
    output["latest_location"] = tool_results[-1]

  if ai_response and hasattr(ai_response, "content") and ai_response.content:
    output["message"] = ai_response.content

  if not output:
    return {"ok": False, "error": "No response produced. Graph may not have called tools or returned AI content.", "history": msgs}

  return {"ok": True, "output": output, "history": msgs}


# -----------------------------
# Flask App + UI
# -----------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Store conversation history per session
CONVERSATION_HISTORY = {}

# Load environment variables from .env (reload-friendly)
try:
  # Prefer current working directory so editing .env takes effect without restart
  detected_env = find_dotenv(filename=".env", raise_error_if_not_found=False, usecwd=True)
  if detected_env:
    load_dotenv(detected_env, override=True)
  else:
    # Fallback: try repo root based on path structure
    repo_root = Path(__file__).resolve().parents[2]
    fallback_env = repo_root / ".env"
    if fallback_env.exists():
      load_dotenv(fallback_env.as_posix(), override=True)
except Exception:
  # Do not crash if dotenv isn't available or any unexpected error occurs
  pass

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>OSM Navigation (LangGraph)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.min.js"></script>
  <style>
    :root {
      --bg: #ffffff;
      --fg: #111111;
      --muted: #6b7280;
      --primary: #2b7cff;
      --danger: #dc3545;
      --panel: #f9fafb;
      --border: #e5e7eb;
    }
    .dark {
      --bg: #0b1220;
      --fg: #e5e7eb;
      --muted: #9ca3af;
      --primary: #6aa5ff;
      --danger: #ef4444;
      --panel: #0f172a;
      --border: #1f2937;
    }
    html, body { height: 100%; }
    body{font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin:0; padding:0; background: var(--bg); color: var(--fg);}
    #container{max-width:1000px;margin:16px auto;padding:16px;border:1px solid var(--border);border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.06);background: var(--panel)}
    #mapContainer{width:100%;height:560px;margin-top:12px;border-radius:10px;overflow:hidden;border:1px solid var(--border)}
    #query{flex: 1 1 auto; padding:10px;border:1px solid var(--border);border-radius:8px;background: var(--bg);color: var(--fg)}
    button{padding:10px 14px;border:1px solid transparent;border-radius:8px;background:var(--primary);color:#fff;cursor:pointer}
    button:hover{filter: brightness(0.95)}
    .row{display:flex;gap:8px;align-items:center}
    .hint{color:var(--muted);font-size:14px}
    pre{background:var(--bg);padding:10px;border-radius:8px;white-space:pre-wrap;word-wrap:break-word;border:1px solid var(--border)}
    .coords{font-size:12px;color:var(--muted)}

    /* Navigation banner */
    #navBanner { display:none; position: sticky; top: 0; z-index: 500; background: var(--bg); color: var(--fg); border: 1px solid var(--border); border-radius: 10px; padding: 10px; margin: 8px 0; }
    #navTop { display:flex; align-items:center; justify-content: space-between; gap: 8px; }
    #navMain { font-weight: 600; }
    #navMeta { font-size: 12px; color: var(--muted); }
    #navControls button { padding: 6px 10px; border-radius: 6px; }
    #endBtn { background: var(--danger); }
    #muteBtn { background: #64748b; }
    #themeBtn { background: #0ea5e9; }
    #recenterBtn { background: #16a34a; }
    /* Nav arrow icon */
    .nav-arrow-wrapper { width: 42px; height: 42px; transform-origin: 50% 50%; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.35)); }
    .nav-arrow-svg { width: 42px; height: 42px; display:block; }
  </style>
</head>
<body>
  </style>
  <script>
    let user = { lat: 52.5200, lon: 13.4050 };
    let sessionId = 'session_' + Date.now();  // Unique session for conversation
    let conversationHistory = [];

    async function useBrowserLocation(){
      const locBtn = document.getElementById("useLocationBtn");
      if (locBtn) locBtn.textContent = "Getting location...";
      
      console.log("useBrowserLocation called");
      
      if (!navigator.geolocation) { 
        console.log("Browser geolocation not supported, trying IP fallback");
        // Fallback to IP-based geolocation if browser geolocation not supported
        await tryIpGeolocation(locBtn);
        return;
      }
      
      // Try with high accuracy first, shorter timeout
      navigator.geolocation.getCurrentPosition(
        pos => {
          console.log("High-accuracy GPS success:", pos.coords);
          const newLat = pos.coords.latitude; const newLon = pos.coords.longitude;
          if (typeof pos.coords.heading === 'number' && !isNaN(pos.coords.heading)) {
            userHeading = pos.coords.heading;
          } else if (lastUserPos) {
            userHeading = computeBearing(lastUserPos.lat, lastUserPos.lon, newLat, newLon);
          }
          user.lat = newLat; user.lon = newLon;
          lastUserPos = { lat: newLat, lon: newLon };
          document.getElementById("loc").textContent = user.lat.toFixed(5) + ", " + user.lon.toFixed(5);
          updateUserMarker();
          map.setView([user.lat, user.lon], 14);
          if (locBtn) locBtn.textContent = "Use my location ✓";
        },
        err => {
          console.warn("High-accuracy failed:", err.code, err.message);
          // Fallback: try without high accuracy and longer timeout
          navigator.geolocation.getCurrentPosition(
            pos => {
              console.log("Low-accuracy position success:", pos.coords);
              const newLat = pos.coords.latitude; const newLon = pos.coords.longitude;
              if (typeof pos.coords.heading === 'number' && !isNaN(pos.coords.heading)) {
                userHeading = pos.coords.heading;
              } else if (lastUserPos) {
                userHeading = computeBearing(lastUserPos.lat, lastUserPos.lon, newLat, newLon);
              }
              user.lat = newLat; user.lon = newLon;
              lastUserPos = { lat: newLat, lon: newLon };
              document.getElementById("loc").textContent = user.lat.toFixed(5) + ", " + user.lon.toFixed(5);
              updateUserMarker();
              map.setView([user.lat, user.lon], 14);
              if (locBtn) locBtn.textContent = "Use my location ✓";
            },
            async err2 => {
              console.warn("Low-accuracy also failed:", err2.code, err2.message);
              // Final fallback: IP-based geolocation (works on desktops without GPS)
              await tryIpGeolocation(locBtn);
            },
            { enableHighAccuracy: false, timeout: 15000, maximumAge: 60000 }
          );
        },
        { enableHighAccuracy: true, timeout: 5000, maximumAge: 10000 }
      );
    }
    
    async function tryIpGeolocation(locBtn) {
      // Try multiple free IP geolocation services in order
      const services = [
        {
          name: 'ip-api.com',
          url: 'http://ip-api.com/json/',
          parse: function(data) { 
            if (data.status === 'success') {
              return {lat: data.lat, lon: data.lon, city: data.city, country: data.country};
            }
            return null;
          }
        },
        {
          name: 'ipapi.co',
          url: 'https://ipapi.co/json/',
          parse: function(data) {
            if (data.latitude) {
              return {lat: data.latitude, lon: data.longitude, city: data.city, country: data.country_name};
            }
            return null;
          }
        },
        {
          name: 'ipwhois.app',
          url: 'https://ipwhois.app/json/',
          parse: function(data) {
            if (data.success) {
              return {lat: data.latitude, lon: data.longitude, city: data.city, country: data.country};
            }
            return null;
          }
        }
      ];

      for (const service of services) {
        try {
          console.log("Trying " + service.name + "...");
          const response = await fetch(service.url);
          const data = await response.json();
          console.log(service.name + " response:", data);
          
          const location = service.parse(data);
          if (location && location.lat && location.lon) {
            user.lat = parseFloat(location.lat);
            user.lon = parseFloat(location.lon);
            const locationText = user.lat.toFixed(5) + ", " + user.lon.toFixed(5) + " (" + (location.city || 'Unknown') + ", " + (location.country || '') + ")";
            document.getElementById("loc").textContent = locationText;
            updateUserMarker();
            map.setView([user.lat, user.lon], 12);
            if (locBtn) locBtn.textContent = "Use my location ✓";
            console.log("✓ Using IP-based location from " + service.name + ":", location.city, location.country, user.lat, user.lon);
            alert("Location detected via IP:\\n" + location.city + ", " + location.country + "\\n\\nNote: IP-based location is approximate (city-level accuracy).");
            return; // Success!
          }
        } catch (err) {
          console.warn(service.name + " failed:", err);
          continue; // Try next service
        }
      }
      
      // All services failed
      console.error("All IP geolocation services failed");
      if (locBtn) locBtn.textContent = "Use my location";
      alert("Could not determine your location automatically.\n\nAll location services failed. The app will use the default location (Berlin).\n\nYou can still use navigation by asking the agent to find places or navigate to specific addresses.");
    }
  </script>
</head>
<body>
  <div id="container">
    <h2>OSM Navigation (LangGraph)</h2>
    <p class="hint">Ask anything: "Find a hospital nearby" • "Take me to Brandenburg Gate" • "Navigate to Starbucks"</p>
    <div class="row">
      <input type="text" id="query" placeholder="Type your request here..." />
      <button onclick="sendQuery()">Submit</button>
      <button onclick="clearChat()" style="background:var(--danger)">Clear Chat</button>
      <button id="themeToggle" onclick="toggleTheme()" title="Toggle theme">Theme</button>
    </div>

    <!-- Navigation Banner -->
    <div id="navBanner">
      <div id="navTop">
        <div>
          <div id="navMain">Starting navigation…</div>
          <div id="navMeta">—</div>
        </div>
        <div id="navControls" class="row">
          <button id="recenterBtn" onclick="recenterMap()">Recenter</button>
          <button id="muteBtn" onclick="toggleMute()">Mute</button>
          <button id="endBtn" onclick="stopNavigation()">End</button>
        </div>
      </div>
    </div>

    <div class="coords">
      Using location: <span id="loc">52.5200, 13.4050</span>
      <button id="useLocationBtn">Use my location</button>
    </div>
    <h3>Conversation</h3>
    <pre id="agent_response">Start a conversation…</pre>
    <div id="mapContainer"></div>
  </div>

  <script>
    // Global variables
    let user = { lat: 52.5200, lon: 13.4050 };
    let sessionId = 'session_' + Date.now();
    let conversationHistory = [];
    
    // Base map layers for light/dark
    const lightTiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap contributors' });
    const darkTiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap & Carto' });
    let map = L.map('mapContainer', { layers: [lightTiles] }).setView([52.52, 13.405], 13);

  let marker;
  let userMarker;
  let userHeading = 0; // degrees
  let lastUserPos = null; // {lat, lon}
    let routingControl = null;
    let navActive = false;
    let watchId = null;
    let muted = false;
    let darkMode = false;

    function computeBearing(lat1, lon1, lat2, lon2) {
      // Returns bearing in degrees from (lat1,lon1) to (lat2,lon2)
      const toRad = function(deg){ return deg * Math.PI / 180; };
      const toDeg = function(rad){ return rad * 180 / Math.PI; };
      const phi1 = toRad(lat1);
      const phi2 = toRad(lat2);
      const dLambda = toRad(lon2 - lon1);
      const y = Math.sin(dLambda) * Math.cos(phi2);
      const x = Math.cos(phi1)*Math.cos(phi2)*Math.cos(dLambda) + Math.sin(phi1)*Math.sin(phi2)*(-1);
      let theta = Math.atan2(y, Math.cos(phi1)*Math.sin(phi2) - Math.sin(phi1)*Math.cos(phi2)*Math.cos(dLambda));
      let brng = (toDeg(theta) + 360) % 360;
      if (isNaN(brng)) brng = 0;
      return brng;
    }

    function navArrowIcon(deg) {
      // Simple upward chevron/arrow that we rotate by deg
      const rotateStyle = "transform: rotate(" + Math.round(deg) + "deg);";
      const html = '<div class="nav-arrow-wrapper" style="' + rotateStyle + '">' +
                   '<svg class="nav-arrow-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">' +
                   '<path d="M12 2 L19 20 L12 16 L5 20 Z" fill="#2b7cff" stroke="#0b3a99" stroke-width="1" />' +
                   '</svg>' +
                   '</div>';
      return L.divIcon({ className: '', html: html, iconSize: [42,42], iconAnchor: [21,21] });
    }

    function updateUserMarker() {
      if (userMarker) { map.removeLayer(userMarker); }
      userMarker = L.marker([user.lat, user.lon], { icon: navArrowIcon(userHeading), interactive: false }).addTo(map);
    }
    updateUserMarker();

    function speak(text) {
      try {
        if (muted || !('speechSynthesis' in window)) return;
        const u = new SpeechSynthesisUtterance(text);
        u.lang = 'en-US';
        u.rate = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
      } catch (e) { /* no-op */ }
    }

    function toggleMute() {
      muted = !muted;
      document.getElementById('muteBtn').textContent = muted ? 'Unmute' : 'Mute';
    }

    function toggleTheme() {
      darkMode = !darkMode;
      document.documentElement.classList.toggle('dark', darkMode);
      if (darkMode) {
        map.removeLayer(lightTiles); map.addLayer(darkTiles);
      } else {
        map.removeLayer(darkTiles); map.addLayer(lightTiles);
      }
    }

    function recenterMap() {
      map.setView([user.lat, user.lon], Math.max(map.getZoom(), 15));
    }

    function updateBanner(mainText, metaText) {
      const banner = document.getElementById('navBanner');
      document.getElementById('navMain').textContent = mainText || 'Navigating…';
      document.getElementById('navMeta').textContent = metaText || '';
      banner.style.display = 'block';
    }

    function clearBanner() {
      const banner = document.getElementById('navBanner');
      banner.style.display = 'none';
      document.getElementById('navMain').textContent = '';
      document.getElementById('navMeta').textContent = '';
    }

    function stopNavigation() {
      if (routingControl) { map.removeControl(routingControl); routingControl = null; }
      if (watchId) { navigator.geolocation.clearWatch(watchId); watchId = null; }
      navActive = false;
      clearBanner();
      speak('Navigation ended.');
    }

    function startNavigation(destLat, destLon, destLabel) {
      // Remove existing route if any
      if (routingControl) { map.removeControl(routingControl); routingControl = null; }

      // Create routing control with OSRM demo server
      routingControl = L.Routing.control({
        waypoints: [ L.latLng(user.lat, user.lon), L.latLng(destLat, destLon) ],
        router: L.Routing.osrmv1({ serviceUrl: 'https://router.project-osrm.org/route/v1' }),
        routeWhileDragging: false,
        addWaypoints: false,
        draggableWaypoints: false,
        show: false,
        lineOptions: { styles: [{ color: '#2b7cff', opacity: 0.9, weight: 6 }] },
        createMarker: function(i, wp, nWps) {
          // Skip creating a start marker to avoid duplication with our user arrow marker
          if (i === 0) return null;
          const label = destLabel || 'Destination';
          return L.marker(wp.latLng).bindPopup(label);
        }
      }).addTo(map);

      // Fit bounds after route is found and update banner
      routingControl.on('routesfound', function(e) {
        const route = e.routes && e.routes[0];
        if (!route) return;
        try { map.fitBounds(L.latLngBounds(route.coordinates)); } catch (err) {}

        const timeMin = Math.round((route.summary.totalTime || 0) / 60);
        const distKm = (route.summary.totalDistance || 0) / 1000;

        // Find next instruction if available
        let nextText = 'Proceed to your destination';
        if (route.instructions && route.instructions.length > 0) {
          const instr = route.instructions[0];
          nextText = instr.text || nextText;
        }
        const meta = `${distKm.toFixed(1)} km • ${timeMin} min`;
        updateBanner(nextText, meta);
        speak(nextText);
      });

      routingControl.on('routingerror', function(e){
        updateBanner('Routing error. Trying again…', 'Check network/OSRM availability');
      });

      // Watch GPS and re-route as we move
      if (watchId) { navigator.geolocation.clearWatch(watchId); watchId = null; }
      if (navigator.geolocation) {
        watchId = navigator.geolocation.watchPosition(
          pos => {
            const newLat = pos.coords.latitude; const newLon = pos.coords.longitude;
            // Prefer device heading if available; otherwise compute from movement
            if (typeof pos.coords.heading === 'number' && !isNaN(pos.coords.heading)) {
              userHeading = pos.coords.heading;
            } else if (lastUserPos && (Math.abs(newLat - lastUserPos.lat) + Math.abs(newLon - lastUserPos.lon) > 1e-6)) {
              userHeading = computeBearing(lastUserPos.lat, lastUserPos.lon, newLat, newLon);
            }
            user.lat = newLat; user.lon = newLon;
            lastUserPos = { lat: newLat, lon: newLon };
            updateUserMarker();
            try { routingControl.spliceWaypoints(0, 1, L.latLng(user.lat, user.lon)); } catch (err) {}
          },
          err => { /* ignore GPS errors during nav */ },
          { enableHighAccuracy: true, maximumAge: 3000, timeout: 8000 }
        );
      }

      navActive = true;
    }

    // Geolocation functions
    async function useBrowserLocation(){
      const locBtn = document.getElementById("useLocationBtn");
      if (locBtn) locBtn.textContent = "Getting location...";
      
      console.log("useBrowserLocation called");
      
      if (!navigator.geolocation) { 
        console.log("Browser geolocation not supported, trying IP fallback");
        await tryIpGeolocation(locBtn);
        return;
      }
      
      navigator.geolocation.getCurrentPosition(
        pos => {
          console.log("High-accuracy GPS success:", pos.coords);
          user.lat = pos.coords.latitude;
          user.lon = pos.coords.longitude;
          document.getElementById("loc").textContent = user.lat.toFixed(5) + ", " + user.lon.toFixed(5);
          updateUserMarker();
          map.setView([user.lat, user.lon], 14);
          if (locBtn) locBtn.textContent = "Use my location ✓";
        },
        err => {
          console.warn("High-accuracy failed:", err.code, err.message);
          navigator.geolocation.getCurrentPosition(
            pos => {
              console.log("Low-accuracy position success:", pos.coords);
              user.lat = pos.coords.latitude;
              user.lon = pos.coords.longitude;
              document.getElementById("loc").textContent = user.lat.toFixed(5) + ", " + user.lon.toFixed(5);
              updateUserMarker();
              map.setView([user.lat, user.lon], 14);
              if (locBtn) locBtn.textContent = "Use my location ✓";
            },
            async err2 => {
              console.warn("Low-accuracy also failed:", err2.code, err2.message);
              await tryIpGeolocation(locBtn);
            },
            { enableHighAccuracy: false, timeout: 15000, maximumAge: 60000 }
          );
        },
        { enableHighAccuracy: true, timeout: 5000, maximumAge: 10000 }
      );
    }
    
    async function tryIpGeolocation(locBtn) {
      const services = [
        {
          name: 'ip-api.com',
          url: 'http://ip-api.com/json/',
          parse: function(data) { 
            if (data.status === 'success') {
              return {lat: data.lat, lon: data.lon, city: data.city, country: data.country};
            }
            return null;
          }
        },
        {
          name: 'ipapi.co',
          url: 'https://ipapi.co/json/',
          parse: function(data) {
            if (data.latitude) {
              return {lat: data.latitude, lon: data.longitude, city: data.city, country: data.country_name};
            }
            return null;
          }
        },
        {
          name: 'ipwhois.app',
          url: 'https://ipwhois.app/json/',
          parse: function(data) {
            if (data.success) {
              return {lat: data.latitude, lon: data.longitude, city: data.city, country: data.country};
            }
            return null;
          }
        }
      ];

      for (const service of services) {
        try {
          console.log("Trying " + service.name + "...");
          const response = await fetch(service.url);
          const data = await response.json();
          console.log(service.name + " response:", data);
          
          const location = service.parse(data);
          if (location && location.lat && location.lon) {
            user.lat = parseFloat(location.lat);
            user.lon = parseFloat(location.lon);
            const locationText = user.lat.toFixed(5) + ", " + user.lon.toFixed(5) + " (" + (location.city || 'Unknown') + ", " + (location.country || '') + ")";
            document.getElementById("loc").textContent = locationText;
            updateUserMarker();
            map.setView([user.lat, user.lon], 12);
            if (locBtn) locBtn.textContent = "Use my location ✓";
            console.log("✓ Using IP-based location from " + service.name + ":", location.city, location.country, user.lat, user.lon);
            alert("Location detected via IP:\\n" + location.city + ", " + location.country + "\\n\\nNote: IP-based location is approximate (city-level accuracy).");
            return;
          }
        } catch (err) {
          console.warn(service.name + " failed:", err);
          continue;
        }
      }
      
      console.error("All IP geolocation services failed");
      if (locBtn) locBtn.textContent = "Use my location";
      alert("Could not determine your location automatically.\\n\\nAll location services failed. The app will use the default location (Berlin).\\n\\nYou can still use navigation by asking the agent to find places or navigate to specific addresses.");
    }

    async function sendQuery(){
      const q = document.getElementById("query").value;
      if (!q.trim()) return;
      
      const out = document.getElementById("agent_response");
      
      // Add user message to display
      conversationHistory.push({role: 'user', text: q});
      displayConversation();
      
      // Clear input
      document.getElementById("query").value = "";
      out.textContent = "Thinking...";
      
      try {
        const res = await fetch("/agent_query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            query: q, 
            user_lat: user.lat, 
            user_lon: user.lon,
            session_id: sessionId
          })
        });
        const data = await res.json();
        
        if (data.ok && data.output) {
          const output = data.output;
          
          if (output.message) {
            conversationHistory.push({role: 'agent', text: output.message});
          }

          if (output.latest_location) {
            const loc = output.latest_location;
            if (loc.lat && loc.lon) {
              // If this is a POI result, just place a marker
              if (loc.type === 'nearest_place') {
                if (marker) map.removeLayer(marker);
                const label = `${loc.place_type}: ${loc.address} (${loc.distance || ''})`;
                marker = L.marker([loc.lat, loc.lon]).addTo(map).bindPopup(label).openPopup();
                map.setView([loc.lat, loc.lon], 15);
              }
              // If this is navigation coordinates, start routing
              if (loc.type === 'navigation_coords') {
                const label = `Navigate to: ${loc.address || 'Destination'}`;
                startNavigation(loc.lat, loc.lon, label);
              }
            }
          }
        } else {
          conversationHistory.push({role: 'agent', text: 'Error: ' + (data.error || 'Unknown error')});
        }
        
        displayConversation();
      } catch (err) {
        conversationHistory.push({role: 'agent', text: 'Error: ' + err});
        displayConversation();
      }
    }
    
    function displayConversation() {
      const out = document.getElementById("agent_response");
      let html = '';
      conversationHistory.forEach(msg => {
        const style = msg.role === 'user' 
          ? 'color: var(--primary); font-weight: bold;' 
          : 'color: var(--fg);';
        html += `<div style="${style}margin-bottom:8px"><strong>${msg.role === 'user' ? 'You' : 'Agent'}:</strong> ${msg.text}</div>`;
      });
      out.innerHTML = html || 'Start a conversation...';
    }
    
    function clearChat() {
      conversationHistory = [];
      sessionId = 'session_' + Date.now();
      displayConversation();
      if (marker) map.removeLayer(marker);
      stopNavigation();
    }
    
    // Allow Enter key to send
    document.addEventListener('DOMContentLoaded', function() {
      document.getElementById("query").addEventListener("keypress", function(e) {
        if (e.key === "Enter") sendQuery();
      });
      
      // Attach location button handler
      document.getElementById("useLocationBtn").addEventListener("click", function() {
        useBrowserLocation();
      });
      
      displayConversation();
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/agent_query", methods=["POST"])
def agent_query():
    payload = request.get_json(force=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"ok": False, "error": "Query cannot be empty"}), 400

    user_lat = payload.get("user_lat")
    user_lon = payload.get("user_lon")
    session_id = payload.get("session_id", "default")  # Track conversation by session

    # Check for LLM configuration (allow any provider)
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY"))
    has_ollama = bool(os.getenv("USE_OLLAMA"))
    
    if not (has_openai or has_gemini or has_ollama):
        return jsonify({
            "ok": False,
            "error": "No LLM configured",
            "hint": "Add to .env: GOOGLE_API_KEY (free at https://makersuite.google.com/app/apikey) or OPENAI_API_KEY",
        }), 500

    try:
        # Get conversation history for this session
        history = CONVERSATION_HISTORY.get(session_id, [])
        
        # Run agent with history
        result = run_agent(query, user_lat=user_lat, user_lon=user_lon, conversation_history=history)
        
        # Update conversation history (store the message objects, not the result dict)
        if result.get("ok") and result.get("history"):
            CONVERSATION_HISTORY[session_id] = result["history"]
        
        # Convert result to JSON-serializable format
        if result.get("ok"):
            output = result.get("output", {})
            return jsonify({
                "ok": True,
                "output": output
            })
        else:
            return jsonify(result)
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in agent_query: {error_trace}")
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Windows PowerShell example to run:
    #   $env:OPENAI_API_KEY="sk-..."; python .\navigation.py
    app.run(host="127.0.0.1", port=5000, debug=True)