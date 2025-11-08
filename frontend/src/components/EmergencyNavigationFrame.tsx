import React, { useCallback, useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-routing-machine/dist/leaflet-routing-machine.css";
import { useDashboardStore } from "../store/dashboard";

type DecisionStep = "hospital" | "pullover" | "navigating" | null;

const NAVIGATION_ENDPOINT =
  (import.meta.env.VITE_NAVIGATION_API_URL as string | undefined) || "/navigation/agent";

const resolveEndpoint = () => {
  if (NAVIGATION_ENDPOINT.startsWith("http")) {
    return NAVIGATION_ENDPOINT;
  }
  if (typeof window !== "undefined") {
    return `${window.location.origin}${NAVIGATION_ENDPOINT}`;
  }
  return NAVIGATION_ENDPOINT;
};

const playBeep = () => {
  try {
    if (typeof window === "undefined") {
      return;
    }
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();
    oscillator.type = "sine";
    oscillator.frequency.value = 880;
    gain.gain.value = 0.2;
    oscillator.connect(gain);
    gain.connect(ctx.destination);
    const now = ctx.currentTime;
    oscillator.start(now);
    oscillator.stop(now + 0.25);
  } catch (err) {
    // Ignore audio issues (autoplay policies, missing API)
  }
};

export const EmergencyNavigationFrame: React.FC = () => {
  const visible = useDashboardStore((state) => state.navigationVisible);
  const navigationLoading = useDashboardStore((state) => state.navigationLoading);
  const setNavigationLoading = useDashboardStore((state) => state.setNavigationLoading);
  const setNavigationDestination = useDashboardStore((state) => state.setNavigationDestination);
  const resetSevereArrhythmiaCount = useDashboardStore((state) => state.resetSevereArrhythmiaCount);
  const silenceNavigation = useDashboardStore((state) => state.silenceNavigation);

  const [currentStep, setCurrentStep] = useState<DecisionStep>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [latestPlace, setLatestPlace] = useState<
    { place_type?: string; address?: string; distance?: string } | null
  >(null);
  const [navigationStarted, setNavigationStarted] = useState(false);
  const [latestCoords, setLatestCoords] = useState<
    | {
        lat: number;
        lon: number;
        address?: string;
      }
    | null
  >(null);
  const coordsRef = useRef<{ lat: number; lon: number } | null>(null);
  const sessionRef = useRef<string>("");
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const routingRef = useRef<any>(null);
  const markersRef = useRef<L.LayerGroup | null>(null);
  const routingLoadedRef = useRef(false);
  const locationRequestedRef = useRef(false);
  const userMarkerRef = useRef<L.Marker | null>(null);
  const userHeadingRef = useRef<number>(0);
  const lastUserPosRef = useRef<{ lat: number; lon: number } | null>(null);
  const watchIdRef = useRef<number | null>(null);
  const [followUser, setFollowUser] = useState<boolean>(true);
  const [nextTurn, setNextTurn] = useState<string | null>(null);
  const [routeSummary, setRouteSummary] = useState<{ distanceKm: number; timeMin: number } | null>(null);
  const lastSpokenStepRef = useRef<DecisionStep>(null);

  // Simple TTS helper using the Speech Synthesis API
  const speakText = useCallback((text: string) => {
    try {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1.0;
      utter.pitch = 1.0;
      utter.volume = 1.0;
      window.speechSynthesis.speak(utter);
    } catch {
      // ignore speech errors
    }
  }, []);

  const ensureRoutingMachineLoaded = useCallback(async () => {
    if (routingLoadedRef.current) {
      return true;
    }
    if (typeof window === "undefined") {
      return false;
    }
    (window as any).L = L;
    try {
      await import("leaflet-routing-machine");
      routingLoadedRef.current = true;
      return true;
    } catch {
      return false;
    }
  }, []);

  // Bearing helper: returns heading degrees from (lat1,lon1) to (lat2,lon2)
  const computeBearing = useCallback((lat1: number, lon1: number, lat2: number, lon2: number) => {
    const toRad = (deg: number) => (deg * Math.PI) / 180;
    const toDeg = (rad: number) => (rad * 180) / Math.PI;
    const phi1 = toRad(lat1);
    const phi2 = toRad(lat2);
    const dLambda = toRad(lon2 - lon1);
    const y = Math.sin(dLambda) * Math.cos(phi2);
    const x = Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(dLambda);
    let theta = Math.atan2(y, x);
    let brng = (toDeg(theta) + 360) % 360;
    if (isNaN(brng)) brng = 0;
    return brng;
  }, []);

  // Build a rotated arrow icon
  const navArrowIcon = useCallback((deg: number) => {
    const rotateStyle = `transform: rotate(${Math.round(deg)}deg);`;
    const html = `
      <div class="nav-arrow-wrapper" style="${rotateStyle} width:42px;height:42px;">
        <svg class="nav-arrow-svg" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" style="width:42px;height:42px;display:block;">
          <path d="M12 2 L19 20 L12 16 L5 20 Z" fill="#2b7cff" stroke="#0b3a99" stroke-width="1" />
        </svg>
      </div>`;
    return L.divIcon({ className: "", html, iconSize: [42, 42], iconAnchor: [21, 21] });
  }, []);

  const tryIpGeolocation = useCallback(async () => {
    // Attempting approximate location via network...
    const services = [
      {
        name: "ipapi.co",
        url: "https://ipapi.co/json/",
        parse: (data: any) =>
          data && typeof data.latitude === "number"
            ? { lat: data.latitude, lon: data.longitude, city: data.city, country: data.country_name }
            : null,
      },
      {
        name: "ipwhois",
        url: "https://ipwhois.app/json/",
        parse: (data: any) =>
          data && data.success
            ? { lat: Number(data.latitude), lon: Number(data.longitude), city: data.city, country: data.country }
            : null,
      },
      {
        name: "ip-api",
        url: "https://ip-api.com/json/",
        parse: (data: any) =>
          data && data.status === "success"
            ? { lat: data.lat, lon: data.lon, city: data.city, country: data.country }
            : null,
      },
    ];

    for (const service of services) {
      try {
        const response = await fetch(service.url, { cache: "no-store" });
        if (!response.ok) continue;
        const data = await response.json();
        const location = service.parse(data);
        if (location && typeof location.lat === "number" && typeof location.lon === "number") {
          coordsRef.current = { lat: location.lat, lon: location.lon };
          // Approximate location set from network
          return true;
        }
      } catch {
        // try next service
      }
    }

    // Could not determine location automatically
    return false;
  }, []);

  const resetState = useCallback(() => {
    setCurrentStep(null);
    setStatusMessage("");
    setLatestPlace(null);
    setNavigationStarted(false);
    setLatestCoords(null);
    setNavigationDestination(undefined);
    setNavigationLoading(false);
    coordsRef.current = null;
    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
    }
    routingRef.current = null;
    markersRef.current = null;
  }, [setNavigationDestination, setNavigationLoading]);

  const requestLocation = useCallback(async () => {
    if (typeof window === "undefined") {
      return;
    }

    const secureEnough =
      window.location.protocol === "https:" ||
      window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1";

    if (typeof navigator === "undefined" || !navigator.geolocation) {
      await tryIpGeolocation();
      return;
    }

    if (!secureEnough) {
      await tryIpGeolocation();
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        coordsRef.current = { lat: pos.coords.latitude, lon: pos.coords.longitude };
      },
      async (err) => {
        await tryIpGeolocation();
      },
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 }
    );
  }, [tryIpGeolocation]);

  // Wait for user location to be available (polls up to timeoutMs)
  const waitForLocation = useCallback(async (timeoutMs: number = 6000) => {
    const start = Date.now();
    if (!coordsRef.current) {
      // Kick off a location request if not already running
      void requestLocation();
    }
    while (!coordsRef.current && Date.now() - start < timeoutMs) {
      await new Promise((r) => setTimeout(r, 200));
    }
    return !!coordsRef.current;
  }, [requestLocation]);

  useEffect(() => {
    if (visible) {
      sessionRef.current = `nav_${Date.now()}`;
      playBeep();
      setCurrentStep("hospital");
      setStatusMessage("");
      setNavigationDestination(undefined);
      setNavigationStarted(false);
      setLatestCoords(null);
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
        routingRef.current = null;
        markersRef.current = null;
      }
      // Auto-detect location when dialog opens
      if (!locationRequestedRef.current) {
        locationRequestedRef.current = true;
        void requestLocation();
      }

      // Speak the first question on open
      setTimeout(() => {
        const q = "Do you want me to find the nearest hospital?";
        speakText(q);
        lastSpokenStepRef.current = "hospital";
      }, 200);
    } else {
      resetState();
      locationRequestedRef.current = false;
    }
  }, [resetState, setNavigationDestination, visible, requestLocation, speakText]);

  // When step changes, read out the question once
  useEffect(() => {
    if (!visible) return;
    if (currentStep === "hospital" && lastSpokenStepRef.current !== "hospital") {
      speakText("Do you want me to find the nearest hospital?");
      lastSpokenStepRef.current = "hospital";
    }
    if (currentStep === "pullover" && lastSpokenStepRef.current !== "pullover") {
      speakText("Do you want to pull over to the next nearest stop?");
      lastSpokenStepRef.current = "pullover";
    }
  }, [currentStep, speakText, visible]);

  const handleClose = useCallback(() => {
    silenceNavigation();
    resetSevereArrhythmiaCount();
    resetState();
  }, [resetSevereArrhythmiaCount, resetState, silenceNavigation]);

  const handleDecision = useCallback(
    async (decision: "hospital" | "pullover", answer: "yes" | "no") => {
      if (answer === "no") {
        if (decision === "hospital") {
          setCurrentStep("pullover");
        } else {
          handleClose();
        }
        return;
      }

      // User said YES - automatically navigate
      setCurrentStep("navigating");
      setNavigationLoading(true);
      
        // Ensure we have user coordinates before querying
        const gotLocation = await waitForLocation(7000);

        const query = decision === "hospital"
          ? "Find nearest hospital within 15 km and start navigation immediately"
          : "Find nearest rest stop or gas station within 15 km and start navigation immediately";
      
        setStatusMessage(`Searching for ${decision === "hospital" ? "nearest hospital" : "nearest rest stop"}...`);

        const payload: Record<string, unknown> = {
          query,
          session_id: sessionRef.current || `nav_${Date.now()}`,
        };

      if (coordsRef.current) {
        payload.user_lat = coordsRef.current.lat;
        payload.user_lon = coordsRef.current.lon;
      }

      const endpoint = resolveEndpoint();

      // Retry loop up to 3 times with backoff
      let success = false;
      const maxAttempts = 3;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        setStatusMessage(`Searching (attempt ${attempt}/${maxAttempts})...`);
        try {
          // Include current coords if available
          if (coordsRef.current) {
            payload.user_lat = coordsRef.current.lat;
            payload.user_lon = coordsRef.current.lon;
          }
          // Debug payload in console
          try { console.debug("nav payload", payload); } catch {}

          const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });

          const raw = await res.text();
          if (!res.ok) {
            let detail: string | undefined;
            if (raw) {
              try {
                const parsed = JSON.parse(raw);
                detail = parsed.detail || parsed.error || parsed.message;
                if (parsed.error?.status === 429 || res.status === 429) {
                  detail = "Navigation service is busy. Please try again in a moment.";
                }
              } catch {
                detail = raw.length > 200 ? raw.slice(0, 200) + "..." : raw;
              }
            }
            throw new Error(detail || `HTTP ${res.status}`);
          }

          const data = raw ? JSON.parse(raw) : null;
          if (!data || !data.ok) {
            throw new Error(data?.error || "Navigation failed");
          }

          const output = data.output || {};
          const latest = output.latest_location;

          if (latest && typeof latest === "object") {
            if (latest.type === "nearest_place") {
              setLatestPlace({
                place_type: latest.place_type,
                address: latest.address,
                distance: latest.distance,
              });
              setNavigationDestination({
                name: latest.address?.split(",")[0]?.trim() || latest.place_type || "Destination",
                address: latest.address || "",
              });
              // Automatically start navigation with the found place coordinates
              if (latest.lat && latest.lon) {
                setNavigationStarted(true);
                setLatestCoords({ lat: latest.lat, lon: latest.lon, address: latest.address });
                setStatusMessage(`Navigating to ${latest.address || latest.place_type}...`);
                success = true;
                break;
              }
            }
            if (latest.type === "navigation_coords") {
              setNavigationStarted(true);
              setLatestCoords({ lat: latest.lat, lon: latest.lon, address: latest.address });
              setStatusMessage(`Navigating to ${latest.address || "destination"}...`);
              success = true;
              break;
            }
          }

          // If we got here, treat as retryable (no coords yet)
          if (attempt < maxAttempts) {
            await new Promise((r) => setTimeout(r, 500 * attempt));
          }
        } catch (err: any) {
          const msg = String(err?.message || err || "Navigation failed");
          // Retry on 5xx/429/timeouts
          const retryable = /HTTP 5\d\d|429|timed out|timeout|temporarily busy|Service Unavailable/i.test(msg);
          if (attempt < maxAttempts && retryable) {
            await new Promise((r) => setTimeout(r, 700 * attempt));
            continue;
          } else {
            setStatusMessage(`Error: ${msg}`);
          }
        }
      }

      if (!success) {
        setStatusMessage(
          decision === "hospital"
            ? "No nearby hospital found or service busy. Please try again shortly."
            : "No nearby rest stop found or service busy. Please try again shortly."
        );
      }
      setNavigationLoading(false);
    },
    [handleClose, setNavigationDestination, setNavigationLoading]
  );

  useEffect(() => {
    if (!visible) {
      return;
    }

    const container = mapContainerRef.current;
    if (!latestCoords || !container) {
      return;
    }

    let cancelled = false;

    const initMap = async () => {
      const loaded = await ensureRoutingMachineLoaded();
      if (!loaded || cancelled) {
        return;
      }

      if (!mapRef.current) {
        mapRef.current = L.map(container, {
          zoomControl: true,
        });
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: "&copy; OpenStreetMap contributors",
        }).addTo(mapRef.current);
        // Disable follow mode when user interacts with the map
        mapRef.current.on("dragstart", () => setFollowUser(false));
        mapRef.current.on("zoomstart", () => setFollowUser(false));
        mapRef.current.on("mousedown", () => setFollowUser(false));
        mapRef.current.on("touchstart", () => setFollowUser(false));
      } else {
        mapRef.current.invalidateSize();
      }

      const userCoords = coordsRef.current;
      const destLatLng = L.latLng(latestCoords.lat, latestCoords.lon);

      if (routingRef.current && mapRef.current) {
        mapRef.current.removeControl(routingRef.current);
        routingRef.current = null;
      }

      const routingModule = (L as any).Routing;
      if (routingModule && mapRef.current) {
        routingRef.current = routingModule
          .control({
            waypoints: userCoords ? [L.latLng(userCoords.lat, userCoords.lon), destLatLng] : [destLatLng],
            router: routingModule.osrmv1({ serviceUrl: "https://router.project-osrm.org/route/v1" }),
            lineOptions: { styles: [{ color: "#2563eb", weight: 5 }] },
            addWaypoints: false,
            draggableWaypoints: false,
            fitSelectedRoutes: true,
            show: false,
          })
          .addTo(mapRef.current);

        // Listen for route results to extract next turn and summary
        try {
          routingRef.current.on("routesfound", (e: any) => {
            const route = e?.routes?.[0];
            if (route) {
              const dist = route.summary?.totalDistance ?? route.summary?.total_distance ?? 0;
              const time = route.summary?.totalTime ?? route.summary?.total_time ?? 0;
              setRouteSummary({ distanceKm: dist / 1000, timeMin: Math.round(time / 60) });
              let instruction: string | null = null;
              if (Array.isArray(route.instructions) && route.instructions.length > 0) {
                instruction = route.instructions[0]?.text || null;
              } else if (route?.name) {
                instruction = String(route.name);
              }
              setNextTurn(instruction);
            } else {
              setNextTurn(null);
              setRouteSummary(null);
            }
          });
          routingRef.current.on("routingerror", () => {
            setNextTurn(null);
            setRouteSummary(null);
          });
        } catch {}
      }

      if (!markersRef.current && mapRef.current) {
        markersRef.current = L.layerGroup().addTo(mapRef.current);
      }

      if (markersRef.current) {
        markersRef.current.clearLayers();
        const destMarker = L.marker(destLatLng).bindPopup(latestCoords.address || "Destination");
        markersRef.current.addLayer(destMarker);
      }

      // Initialize or update user arrow marker
      if (mapRef.current && coordsRef.current) {
        const { lat, lon } = coordsRef.current;
        if (!userMarkerRef.current) {
          userMarkerRef.current = L.marker([lat, lon], { icon: navArrowIcon(userHeadingRef.current), interactive: false }).addTo(mapRef.current);
        } else {
          userMarkerRef.current.setLatLng([lat, lon]);
          userMarkerRef.current.setIcon(navArrowIcon(userHeadingRef.current));
        }
        lastUserPosRef.current = { lat, lon };
      }

      // Start watching user movement to update marker and route origin
      if (navigator.geolocation && mapRef.current) {
        if (watchIdRef.current != null) {
          try { navigator.geolocation.clearWatch(watchIdRef.current); } catch {}
          watchIdRef.current = null;
        }
        watchIdRef.current = navigator.geolocation.watchPosition(
          (pos) => {
            const newLat = pos.coords.latitude;
            const newLon = pos.coords.longitude;
            let heading = userHeadingRef.current;
            if (typeof pos.coords.heading === "number" && !isNaN(pos.coords.heading)) {
              heading = pos.coords.heading;
            } else if (lastUserPosRef.current) {
              heading = computeBearing(lastUserPosRef.current.lat, lastUserPosRef.current.lon, newLat, newLon);
            }
            userHeadingRef.current = heading;
            coordsRef.current = { lat: newLat, lon: newLon };
            lastUserPosRef.current = { lat: newLat, lon: newLon };
            if (userMarkerRef.current) {
              userMarkerRef.current.setLatLng([newLat, newLon]);
              userMarkerRef.current.setIcon(navArrowIcon(heading));
            } else if (mapRef.current) {
              userMarkerRef.current = L.marker([newLat, newLon], { icon: navArrowIcon(heading), interactive: false }).addTo(mapRef.current);
            }
            try {
              if (routingRef.current) {
                routingRef.current.spliceWaypoints(0, 1, L.latLng(newLat, newLon));
              }
            } catch {
              // ignore routing splice errors
            }

            // If following is enabled, keep map centered on the user at street-level zoom
            if (followUser && mapRef.current) {
              const currentZoom = mapRef.current.getZoom() ?? 13;
              const targetZoom = currentZoom < 16 ? 17 : currentZoom;
              try { mapRef.current.setView([newLat, newLon], targetZoom, { animate: true }); } catch {}
            }
          },
          () => { /* ignore errors while watching */ },
          { enableHighAccuracy: true, maximumAge: 3000, timeout: 8000 }
        );
      }

      if (mapRef.current) {
        if (followUser && coordsRef.current) {
          mapRef.current.setView([coordsRef.current.lat, coordsRef.current.lon], 17);
        } else {
          mapRef.current.setView(destLatLng, 13);
        }
        setTimeout(() => {
          mapRef.current && mapRef.current.invalidateSize();
        }, 50);
      }
    };

    void initMap();

    return () => {
      cancelled = true;
      // Cleanup watch and user marker
      if (watchIdRef.current != null) {
        try { navigator.geolocation.clearWatch(watchIdRef.current); } catch {}
        watchIdRef.current = null;
      }
      // Remove listeners on routing control
      if (routingRef.current) {
        try {
          routingRef.current.off("routesfound");
          routingRef.current.off("routingerror");
        } catch {}
      }
      if (mapRef.current && userMarkerRef.current) {
        try { mapRef.current.removeLayer(userMarkerRef.current); } catch {}
        userMarkerRef.current = null;
      }
    };
  }, [ensureRoutingMachineLoaded, latestCoords, visible, followUser]);

  if (!visible) {
    return null;
  }

  return (
    <div
      className="emergency-overlay navigation-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="navigation-dialog-title"
    >
      <div className="emergency-card navigation-card">
        <header className="navigation-card-header">
          <div className="navigation-card-header-top">
            <span className="navigation-badge">Guardian Navigator</span>
          </div>
          {/* Title removed per request */}
        </header>

        {/* Removed highlight panel to avoid duplicate address/header */}

        {latestCoords ? (
          <section className="navigation-panel navigation-panel-route">
            {/* Minimize duplicate headers/content; show concise coordinates + map */}
            {(nextTurn || routeSummary) && (
              <div className="navigation-turn-banner">
                <div className="navigation-turn-text">{nextTurn || "Route ready"}</div>
                {routeSummary && (
                  <div className="navigation-turn-meta">{routeSummary.distanceKm.toFixed(1)} km • {routeSummary.timeMin} min</div>
                )}
              </div>
            )}
            <div className="navigation-coordinates">
              Lat: {latestCoords.lat.toFixed(5)} | Lon: {latestCoords.lon.toFixed(5)}
            </div>
            <div className="navigation-map-wrapper">
              <div ref={mapContainerRef} className="navigation-map" />
              <button
                type="button"
                className={`navigation-recenter-button${followUser ? " is-following" : ""}`}
                title={followUser ? "Following" : "Recenter"}
                aria-label={followUser ? "Following" : "Recenter"}
                onClick={() => {
                  setFollowUser(true);
                  if (mapRef.current && coordsRef.current) {
                    const currentZoom = mapRef.current.getZoom() ?? 13;
                    const targetZoom = currentZoom < 16 ? 17 : currentZoom;
                    try { mapRef.current.setView([coordsRef.current.lat, coordsRef.current.lon], targetZoom, { animate: true }); } catch {}
                  }
                }}
              >
                {/* Crosshair/target icon */}
                <svg viewBox="0 0 24 24" width="20" height="20" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                  <circle cx="12" cy="12" r="3" fill={followUser ? "#22c55e" : "#e2e8f0"} />
                  <circle cx="12" cy="12" r="8" fill="none" stroke={followUser ? "#22c55e" : "#cbd5f5"} strokeWidth="1.5" />
                  <path d="M12 2v3M12 19v3M2 12h3M19 12h3" stroke={followUser ? "#22c55e" : "#cbd5f5"} strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </button>
            </div>
          </section>
        ) : null}

        {currentStep === "hospital" && (
          <section className="navigation-decision">
            <div className="navigation-decision-question">
              <h3>Do you want me to find the nearest Hospital?</h3>
            </div>
            <div className="navigation-decision-buttons">
              <button
                type="button"
                className="btn btn-primary btn-large"
                onClick={() => handleDecision("hospital", "yes")}
                disabled={navigationLoading}
              >
                Yes
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-large"
                onClick={() => handleDecision("hospital", "no")}
                disabled={navigationLoading}
              >
                No
              </button>
            </div>
          </section>
        )}

        {currentStep === "pullover" && (
          <section className="navigation-decision">
            <div className="navigation-decision-question">
              <h3>Do you want to pull over to the next nearest stop?</h3>
            </div>
            <div className="navigation-decision-buttons">
              <button
                type="button"
                className="btn btn-primary btn-large"
                onClick={() => handleDecision("pullover", "yes")}
                disabled={navigationLoading}
              >
                Yes
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-large"
                onClick={() => handleDecision("pullover", "no")}
                disabled={navigationLoading}
              >
                No
              </button>
            </div>
          </section>
        )}

        {currentStep === "navigating" && statusMessage && !navigationStarted && !/^Navigating to/i.test(statusMessage) && !/^Found:/i.test(statusMessage) && (
          <section className="navigation-status">
            <div className="navigation-status-message">{statusMessage}</div>
          </section>
        )}

        <div className="navigation-actions">
          <button type="button" className="btn btn-ghost navigation-cancel" onClick={handleClose}>
            Cancel
          </button>
        </div>

        {/* Removed bottom footnote to avoid redundant messaging */}
      </div>
    </div>
  );
};
