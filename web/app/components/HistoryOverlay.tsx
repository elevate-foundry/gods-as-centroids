"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import {
  HISTORICAL_EPOCHS,
  HISTORICAL_EVENTS,
  generateProjections,
  type HistoricalEvent,
} from "../lib/historical-data";

interface SimPoint {
  t: number;
  nEff: number;
  entropy: number;
  dominance: number;
}

interface Props {
  simHistory: SimPoint[];
  currentCoercion: number;
  currentMutationRate: number;
  width: number;
  height: number;
}

type MetricKey = "nEff" | "dominance" | "entropy";

const METRIC_CONFIG: Record<
  MetricKey,
  { label: string; color: string; projColor: string; range: [number, number]; unit: string }
> = {
  nEff: {
    label: "Godforms (N_eff)",
    color: "#818cf8",
    projColor: "#6366f1",
    range: [0, 14],
    unit: "",
  },
  dominance: {
    label: "Dominance",
    color: "#f87171",
    projColor: "#ef4444",
    range: [0, 1],
    unit: "%",
  },
  entropy: {
    label: "Entropy",
    color: "#22d3ee",
    projColor: "#06b6d4",
    range: [0, 4],
    unit: "",
  },
};

const EVENT_COLORS: Record<HistoricalEvent["type"], string> = {
  prophet: "#f59e0b",
  war: "#ef4444",
  schism: "#a855f7",
  syncretism: "#06b6d4",
  empire: "#f97316",
  reform: "#10b981",
  decline: "#6b7280",
};

const EVENT_ICONS: Record<HistoricalEvent["type"], string> = {
  prophet: "✦",
  war: "⚔",
  schism: "⫘",
  syncretism: "⊕",
  empire: "♛",
  reform: "⚖",
  decline: "▾",
};

export default function HistoryOverlay({
  simHistory,
  currentCoercion,
  currentMutationRate,
  width,
  height,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [activeMetric, setActiveMetric] = useState<MetricKey>("nEff");
  const [showProjections, setShowProjections] = useState(true);
  const [showEvents, setShowEvents] = useState(true);
  const [showSimOverlay, setShowSimOverlay] = useState(true);
  const [hoveredEvent, setHoveredEvent] = useState<HistoricalEvent | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [hoveredYear, setHoveredYear] = useState<number | null>(null);

  // Timeline range
  const YEAR_MIN = -3200;
  const YEAR_MAX = 2120;
  const YEAR_RANGE = YEAR_MAX - YEAR_MIN;

  // Chart area
  const MARGIN = { top: 30, right: 20, bottom: 50, left: 50 };
  const chartW = width - MARGIN.left - MARGIN.right;
  const chartH = height - MARGIN.top - MARGIN.bottom;

  const mapX = useCallback(
    (year: number) => MARGIN.left + ((year - YEAR_MIN) / YEAR_RANGE) * chartW,
    [chartW]
  );
  const mapY = useCallback(
    (value: number, metric: MetricKey) => {
      const [lo, hi] = METRIC_CONFIG[metric].range;
      return MARGIN.top + chartH - ((value - lo) / (hi - lo)) * chartH;
    },
    [chartH]
  );

  // Sim step → year mapping: scale sim steps to fill 3000 BCE → 2025 CE
  const simToYear = useCallback(
    (step: number) => {
      if (simHistory.length <= 1) return -3000;
      const maxStep = simHistory[simHistory.length - 1]?.t || 1;
      return -3000 + (step / maxStep) * 5025; // -3000 to 2025
    },
    [simHistory]
  );

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    const mc = METRIC_CONFIG[activeMetric];

    // Background
    ctx.fillStyle = "#0d0d20";
    ctx.fillRect(0, 0, width, height);

    // Future zone shading (2025+)
    const futureX = mapX(2025);
    ctx.fillStyle = "rgba(99, 102, 241, 0.04)";
    ctx.fillRect(futureX, MARGIN.top, width - MARGIN.right - futureX, chartH);

    // "PROJECTION" label in future zone
    if (showProjections) {
      ctx.save();
      ctx.fillStyle = "rgba(99, 102, 241, 0.15)";
      ctx.font = "bold 10px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText("PROJECTION →", (futureX + width - MARGIN.right) / 2, MARGIN.top + 16);
      ctx.restore();
    }

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;

    // Horizontal grid
    const [lo, hi] = mc.range;
    const gridSteps = activeMetric === "dominance" ? 5 : activeMetric === "entropy" ? 4 : 7;
    for (let i = 0; i <= gridSteps; i++) {
      const val = lo + (i / gridSteps) * (hi - lo);
      const y = mapY(val, activeMetric);
      ctx.beginPath();
      ctx.moveTo(MARGIN.left, y);
      ctx.lineTo(width - MARGIN.right, y);
      ctx.stroke();

      // Y-axis labels
      ctx.fillStyle = "rgba(255,255,255,0.25)";
      ctx.font = "9px ui-monospace, monospace";
      ctx.textAlign = "right";
      const label =
        activeMetric === "dominance"
          ? `${(val * 100).toFixed(0)}%`
          : val.toFixed(1);
      ctx.fillText(label, MARGIN.left - 6, y + 3);
    }

    // Vertical grid (centuries)
    const centurySteps = [-3000, -2000, -1000, 0, 500, 1000, 1500, 1800, 1900, 2000, 2050, 2100];
    for (const yr of centurySteps) {
      const x = mapX(yr);
      if (x < MARGIN.left || x > width - MARGIN.right) continue;
      ctx.strokeStyle = yr === 0 ? "rgba(255,255,255,0.10)" : "rgba(255,255,255,0.04)";
      ctx.lineWidth = yr === 0 ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(x, MARGIN.top);
      ctx.lineTo(x, MARGIN.top + chartH);
      ctx.stroke();

      // X-axis labels
      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.font = "9px ui-monospace, monospace";
      ctx.textAlign = "center";
      const yearLabel = yr < 0 ? `${Math.abs(yr)} BCE` : yr === 0 ? "0 CE" : `${yr}`;
      ctx.fillText(yearLabel, x, MARGIN.top + chartH + 16);
    }

    // ─── Historical data line ───────────────────────────────────────
    ctx.strokeStyle = mc.color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.setLineDash([]);
    ctx.beginPath();
    let started = false;
    for (const epoch of HISTORICAL_EPOCHS) {
      const x = mapX(epoch.year);
      const y = mapY(epoch[activeMetric], activeMetric);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Data points
    for (const epoch of HISTORICAL_EPOCHS) {
      const x = mapX(epoch.year);
      const y = mapY(epoch[activeMetric], activeMetric);

      ctx.fillStyle = mc.color;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();

      // Subtle glow
      ctx.shadowColor = mc.color;
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // ─── Simulation overlay ─────────────────────────────────────────
    if (showSimOverlay && simHistory.length > 2) {
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 3]);
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      let simStarted = false;

      // Sample every Nth point to avoid overdraw
      const step = Math.max(1, Math.floor(simHistory.length / 300));
      for (let i = 0; i < simHistory.length; i += step) {
        const pt = simHistory[i];
        const x = mapX(simToYear(pt.t));
        const y = mapY(pt[activeMetric], activeMetric);
        if (x < MARGIN.left || x > width - MARGIN.right) continue;
        if (!simStarted) {
          ctx.moveTo(x, y);
          simStarted = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // ─── Future projections ─────────────────────────────────────────
    if (showProjections) {
      const lastEpoch = HISTORICAL_EPOCHS[HISTORICAL_EPOCHS.length - 1];
      // Use latest sim metrics to influence projections when sim has run
      const lastSim = simHistory.length > 1 ? simHistory[simHistory.length - 1] : null;
      const proj = generateProjections(
        lastEpoch.nEff,
        lastEpoch.dominance,
        lastEpoch.entropy,
        currentCoercion,
        currentMutationRate,
        lastSim?.nEff,
        lastSim?.dominance,
        lastSim?.entropy,
        lastSim && lastSim.t > 100 ? 0.5 : 0, // weight sim more as it runs longer
      );

      // Draw projection bands
      const scenarios = [
        { data: proj.optimistic, color: "#10b981", label: "Pluralistic" },
        { data: proj.baseline, color: mc.projColor, label: "Baseline" },
        { data: proj.pessimistic, color: "#ef4444", label: "Convergent" },
      ];

      // Fill between optimistic and pessimistic
      ctx.globalAlpha = 0.08;
      ctx.fillStyle = mc.projColor;
      ctx.beginPath();
      // Top edge (optimistic for nEff/entropy, pessimistic for dominance)
      const topData = activeMetric === "dominance" ? proj.pessimistic : proj.optimistic;
      const botData = activeMetric === "dominance" ? proj.optimistic : proj.pessimistic;

      ctx.moveTo(mapX(2025), mapY(lastEpoch[activeMetric], activeMetric));
      for (const pt of topData) {
        ctx.lineTo(mapX(pt.year), mapY(pt[activeMetric], activeMetric));
      }
      // Bottom edge (reverse)
      for (let i = botData.length - 1; i >= 0; i--) {
        ctx.lineTo(mapX(botData[i].year), mapY(botData[i][activeMetric], activeMetric));
      }
      ctx.lineTo(mapX(2025), mapY(lastEpoch[activeMetric], activeMetric));
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;

      // Draw scenario lines
      for (const scenario of scenarios) {
        ctx.strokeStyle = scenario.color;
        ctx.lineWidth = scenario.label === "Baseline" ? 2 : 1.2;
        ctx.setLineDash(scenario.label === "Baseline" ? [] : [5, 4]);
        ctx.globalAlpha = scenario.label === "Baseline" ? 0.8 : 0.5;
        ctx.beginPath();
        ctx.moveTo(mapX(2025), mapY(lastEpoch[activeMetric], activeMetric));
        for (const pt of scenario.data) {
          ctx.lineTo(mapX(pt.year), mapY(pt[activeMetric], activeMetric));
        }
        ctx.stroke();

        // End label
        const lastPt = scenario.data[scenario.data.length - 1];
        ctx.fillStyle = scenario.color;
        ctx.font = "bold 8px ui-monospace, monospace";
        ctx.textAlign = "left";
        ctx.fillText(
          scenario.label,
          mapX(lastPt.year) + 4,
          mapY(lastPt[activeMetric], activeMetric) + 3
        );
        ctx.globalAlpha = 1;
      }
      ctx.setLineDash([]);
    }

    // ─── Historical events ──────────────────────────────────────────
    if (showEvents) {
      for (const event of HISTORICAL_EVENTS) {
        const x = mapX(event.year);
        if (x < MARGIN.left || x > width - MARGIN.right) continue;

        const color = EVENT_COLORS[event.type];
        const icon = EVENT_ICONS[event.type];

        // Vertical tick
        ctx.strokeStyle = color + "40";
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(x, MARGIN.top + chartH);
        ctx.lineTo(x, MARGIN.top + chartH + 8);
        ctx.stroke();
        ctx.setLineDash([]);

        // Event marker at bottom
        ctx.fillStyle = color;
        ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(icon, x, MARGIN.top + chartH + 30);

        // Highlight hovered event
        if (hoveredEvent === event) {
          ctx.strokeStyle = color + "30";
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.moveTo(x, MARGIN.top);
          ctx.lineTo(x, MARGIN.top + chartH);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }

    // ─── Hover crosshair ────────────────────────────────────────────
    if (hoveredYear !== null) {
      const x = mapX(hoveredYear);
      ctx.strokeStyle = "rgba(255,255,255,0.15)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, MARGIN.top);
      ctx.lineTo(x, MARGIN.top + chartH);
      ctx.stroke();
      ctx.setLineDash([]);

      // Year label
      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.font = "bold 10px ui-monospace, monospace";
      ctx.textAlign = "center";
      const yearStr = hoveredYear < 0 ? `${Math.abs(Math.round(hoveredYear))} BCE` : `${Math.round(hoveredYear)} CE`;
      ctx.fillText(yearStr, x, MARGIN.top - 6);

      // Find closest epoch and show value
      let closestEpoch = HISTORICAL_EPOCHS[0];
      let closestDist = Infinity;
      for (const e of HISTORICAL_EPOCHS) {
        const d = Math.abs(e.year - hoveredYear);
        if (d < closestDist) { closestDist = d; closestEpoch = e; }
      }
      if (closestDist < 200) {
        const val = closestEpoch[activeMetric];
        const y = mapY(val, activeMetric);
        ctx.fillStyle = mc.color;
        ctx.beginPath();
        ctx.arc(mapX(closestEpoch.year), y, 6, 0, Math.PI * 2);
        ctx.fill();

        // Value tooltip
        ctx.fillStyle = "#0d0d20";
        ctx.fillRect(mapX(closestEpoch.year) + 10, y - 22, 90, 32);
        ctx.strokeStyle = mc.color + "40";
        ctx.lineWidth = 1;
        ctx.strokeRect(mapX(closestEpoch.year) + 10, y - 22, 90, 32);
        ctx.fillStyle = mc.color;
        ctx.font = "bold 10px ui-monospace, monospace";
        ctx.textAlign = "left";
        const dispVal = activeMetric === "dominance" ? `${(val * 100).toFixed(0)}%` : val.toFixed(1);
        ctx.fillText(dispVal, mapX(closestEpoch.year) + 16, y - 8);
        ctx.fillStyle = "rgba(255,255,255,0.4)";
        ctx.font = "8px ui-sans-serif, system-ui, sans-serif";
        ctx.fillText(closestEpoch.label, mapX(closestEpoch.year) + 16, y + 4);
      }
    }

    // ─── Title and legend ───────────────────────────────────────────
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.font = "bold 11px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${mc.label} — 5,000 Years of Religious Evolution`, MARGIN.left, 16);

    // Legend
    const legendX = width - MARGIN.right - 200;
    ctx.fillStyle = mc.color;
    ctx.fillRect(legendX, 8, 12, 3);
    ctx.fillStyle = "rgba(255,255,255,0.4)";
    ctx.font = "8px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.fillText("Historical", legendX + 16, 12);

    if (showSimOverlay) {
      ctx.fillStyle = "#fbbf24";
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(legendX + 80, 10);
      ctx.lineTo(legendX + 92, 10);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.fillText("Simulation", legendX + 96, 12);
    }

    if (showProjections) {
      ctx.fillStyle = mc.projColor + "60";
      ctx.fillRect(legendX + 160, 6, 12, 8);
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.fillText("Projected", legendX + 176, 12);
    }

  }, [
    width, height, activeMetric, showProjections, showEvents, showSimOverlay,
    simHistory, currentCoercion, currentMutationRate, hoveredEvent, hoveredYear,
    mapX, mapY, simToYear, chartH, chartW,
  ]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Mouse interaction
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      setMousePos({ x: mx, y: my });

      // Map to year
      if (mx >= MARGIN.left && mx <= width - MARGIN.right && my >= MARGIN.top && my <= MARGIN.top + chartH) {
        const year = YEAR_MIN + ((mx - MARGIN.left) / chartW) * YEAR_RANGE;
        setHoveredYear(year);
      } else {
        setHoveredYear(null);
      }

      // Check event hover
      let found: HistoricalEvent | null = null;
      for (const event of HISTORICAL_EVENTS) {
        const ex = mapX(event.year);
        const ey = MARGIN.top + chartH + 25;
        if (Math.abs(mx - ex) < 10 && Math.abs(my - ey) < 12) {
          found = event;
          break;
        }
      }
      setHoveredEvent(found);
    },
    [width, chartW, chartH, mapX]
  );

  const handleMouseLeave = useCallback(() => {
    setHoveredEvent(null);
    setHoveredYear(null);
    setMousePos(null);
  }, []);

  return (
    <div className="flex flex-col gap-2">
      {/* Controls bar */}
      <div className="flex items-center gap-3 px-2">
        {/* Metric selector */}
        <div className="flex gap-1">
          {(Object.keys(METRIC_CONFIG) as MetricKey[]).map((key) => (
            <button
              key={key}
              onClick={() => setActiveMetric(key)}
              className={`px-2.5 py-1 rounded text-[10px] font-medium transition-all ${
                activeMetric === key
                  ? "text-white border border-white/20 bg-white/10"
                  : "text-white/40 border border-transparent hover:text-white/60"
              }`}
              style={
                activeMetric === key
                  ? { borderColor: METRIC_CONFIG[key].color + "40", color: METRIC_CONFIG[key].color }
                  : undefined
              }
            >
              {METRIC_CONFIG[key].label}
            </button>
          ))}
        </div>

        <div className="flex-1" />

        {/* Toggles */}
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={showSimOverlay}
            onChange={(e) => setShowSimOverlay(e.target.checked)}
            className="w-3 h-3 rounded accent-amber-400"
          />
          <span className="text-[10px] text-white/40">Sim Overlay</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={showProjections}
            onChange={(e) => setShowProjections(e.target.checked)}
            className="w-3 h-3 rounded accent-indigo-400"
          />
          <span className="text-[10px] text-white/40">Projections</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={showEvents}
            onChange={(e) => setShowEvents(e.target.checked)}
            className="w-3 h-3 rounded accent-amber-400"
          />
          <span className="text-[10px] text-white/40">Events</span>
        </label>
      </div>

      {/* Canvas */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="rounded-lg border border-white/5 cursor-crosshair"
          style={{ width, height }}
        />

        {/* Event tooltip */}
        {hoveredEvent && mousePos && (
          <div
            className="absolute pointer-events-none z-10 px-3 py-2 rounded-lg border bg-[#0d0d20]/95 backdrop-blur-sm max-w-[220px]"
            style={{
              left: Math.min(mousePos.x + 12, width - 240),
              top: mousePos.y - 60,
              borderColor: EVENT_COLORS[hoveredEvent.type] + "40",
            }}
          >
            <div className="flex items-center gap-1.5 mb-1">
              <span style={{ color: EVENT_COLORS[hoveredEvent.type] }}>
                {EVENT_ICONS[hoveredEvent.type]}
              </span>
              <span className="text-xs font-semibold text-white/80">
                {hoveredEvent.label}
              </span>
              <span className="text-[10px] text-white/30 ml-auto">
                {hoveredEvent.year < 0
                  ? `${Math.abs(hoveredEvent.year)} BCE`
                  : `${hoveredEvent.year} CE`}
              </span>
            </div>
            <p className="text-[10px] text-white/40 leading-tight">
              {hoveredEvent.description}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
