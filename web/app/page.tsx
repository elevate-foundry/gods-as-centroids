"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { SwarmKernel, DEFAULT_CONFIG, type SimConfig, type Snapshot } from "./lib/simulation";
import SimCanvas from "./components/SimCanvas";
import ControlPanel from "./components/ControlPanel";
import MetricsPanel from "./components/MetricsPanel";
import InfoPanel from "./components/InfoPanel";
import HistoryOverlay from "./components/HistoryOverlay";
import TheologyPanel from "./components/TheologyPanel";

export default function Home() {
  const [config, setConfig] = useState<SimConfig>({ ...DEFAULT_CONFIG });
  const kernelRef = useRef<SwarmKernel | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [history, setHistory] = useState<
    { t: number; nEff: number; entropy: number; dominance: number }[]
  >([]);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(5);
  const runningRef = useRef(false);
  const rafRef = useRef<number>(0);
  const [canvasSize, setCanvasSize] = useState({ w: 700, h: 600 });
  const [timelineWidth, setTimelineWidth] = useState(700);
  const [showTimeline, setShowTimeline] = useState(true);
  const [activeTab, setActiveTab] = useState<"controls" | "info">("controls");
  const [showTheology, setShowTheology] = useState(true);

  // Initialize kernel
  useEffect(() => {
    const kernel = new SwarmKernel(config);
    kernelRef.current = kernel;
    const snap = kernel.snapshot();
    setSnapshot(snap);
    setHistory([{ t: snap.t, nEff: snap.nEff, entropy: snap.entropy, dominance: snap.dominance }]);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Responsive canvas sizing
  useEffect(() => {
    const handleResize = () => {
      const centerW = Math.min(window.innerWidth - 580, 900);
      const w = Math.max(400, centerW);
      const canvasH = showTimeline
        ? Math.min(window.innerHeight - 340, 500)
        : Math.min(window.innerHeight - 120, 700);
      setCanvasSize({ w, h: Math.max(300, canvasH) });
      setTimelineWidth(w);
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [showTimeline]);

  // Animation loop
  const tick = useCallback(() => {
    if (!runningRef.current || !kernelRef.current) return;
    const kernel = kernelRef.current;
    for (let i = 0; i < speed; i++) kernel.step();
    const snap = kernel.snapshot();
    setSnapshot(snap);
    setHistory((prev) => {
      const next = [...prev, { t: snap.t, nEff: snap.nEff, entropy: snap.entropy, dominance: snap.dominance }];
      if (next.length > 500) return next.slice(-500);
      return next;
    });
    rafRef.current = requestAnimationFrame(tick);
  }, [speed]);

  useEffect(() => {
    runningRef.current = running;
    if (running) {
      rafRef.current = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(rafRef.current);
    }
    return () => cancelAnimationFrame(rafRef.current);
  }, [running, tick]);

  const handleConfigChange = useCallback((partial: Partial<SimConfig>) => {
    setConfig((prev) => {
      const next = { ...prev, ...partial };
      if (kernelRef.current) kernelRef.current.updateConfig(partial);
      return next;
    });
  }, []);

  const handleReset = useCallback(() => {
    setRunning(false);
    runningRef.current = false;
    const kernel = new SwarmKernel(config);
    kernelRef.current = kernel;
    const snap = kernel.snapshot();
    setSnapshot(snap);
    setHistory([{ t: snap.t, nEff: snap.nEff, entropy: snap.entropy, dominance: snap.dominance }]);
  }, [config]);

  const handlePreset = useCallback((preset: Partial<SimConfig>) => {
    setRunning(false);
    runningRef.current = false;
    const newConfig = { ...DEFAULT_CONFIG, ...preset };
    setConfig(newConfig);
    const kernel = new SwarmKernel(newConfig);
    kernelRef.current = kernel;
    const snap = kernel.snapshot();
    setSnapshot(snap);
    setHistory([{ t: snap.t, nEff: snap.nEff, entropy: snap.entropy, dominance: snap.dominance }]);
  }, []);

  const handleProphet = useCallback(() => {
    if (!kernelRef.current) return;
    kernelRef.current.injectProphet();
    const snap = kernelRef.current.snapshot();
    setSnapshot(snap);
    setHistory((prev) => [...prev, { t: snap.t, nEff: snap.nEff, entropy: snap.entropy, dominance: snap.dominance }]);
  }, []);

  const handleShock = useCallback((type: "war" | "plague" | "abundance" | "contact") => {
    if (!kernelRef.current) return;
    kernelRef.current.environmentalShock(type);
    const snap = kernelRef.current.snapshot();
    setSnapshot(snap);
    setHistory((prev) => [...prev, { t: snap.t, nEff: snap.nEff, entropy: snap.entropy, dominance: snap.dominance }]);
  }, []);

  const handleSyncretism = useCallback(() => {
    if (!kernelRef.current) return;
    kernelRef.current.forceSyncretism();
    const snap = kernelRef.current.snapshot();
    setSnapshot(snap);
    setHistory((prev) => [...prev, { t: snap.t, nEff: snap.nEff, entropy: snap.entropy, dominance: snap.dominance }]);
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0a1a] text-white flex flex-col">
      {/* Header */}
      <header className="border-b border-white/5 px-6 py-4 flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-lg font-bold tracking-tight">
            Gods as Centroids
          </h1>
          <p className="text-xs text-white/40 mt-0.5">
            A Swarm-Based Vector Model of Religious Evolution
          </p>
        </div>
        <div className="text-xs text-white/20">
          Ryan Barrett &middot; 2025
        </div>
      </header>

      {/* Main layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar: Controls */}
        <aside className="w-[300px] border-r border-white/5 flex flex-col flex-shrink-0">
          {/* Tabs */}
          <div className="flex border-b border-white/5">
            <button
              onClick={() => setActiveTab("controls")}
              className={`flex-1 py-2.5 text-xs font-medium transition-colors ${
                activeTab === "controls"
                  ? "text-white border-b-2 border-indigo-400"
                  : "text-white/40 hover:text-white/60"
              }`}
            >
              Controls
            </button>
            <button
              onClick={() => setActiveTab("info")}
              className={`flex-1 py-2.5 text-xs font-medium transition-colors ${
                activeTab === "info"
                  ? "text-white border-b-2 border-indigo-400"
                  : "text-white/40 hover:text-white/60"
              }`}
            >
              About
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            {activeTab === "controls" ? (
              <ControlPanel
                config={config}
                onChange={handleConfigChange}
                onReset={handleReset}
                onPreset={handlePreset}
                running={running}
                onToggleRun={() => setRunning((r) => !r)}
                speed={speed}
                onSpeedChange={setSpeed}
                onProphet={handleProphet}
                onShock={handleShock}
                onSyncretism={handleSyncretism}
              />
            ) : (
              <InfoPanel />
            )}
          </div>
        </aside>

        {/* Center: Canvas + Timeline */}
        <main className="flex-1 flex flex-col items-center p-4 overflow-y-auto overflow-x-hidden gap-3">
          <SimCanvas
            snapshot={snapshot}
            width={canvasSize.w}
            height={canvasSize.h}
          />

          {/* Timeline toggle + panel */}
          <div className="w-full flex flex-col items-center">
            <button
              onClick={() => setShowTimeline((v) => !v)}
              className="text-[10px] text-white/30 hover:text-white/50 transition-colors mb-1 flex items-center gap-1"
            >
              <span>{showTimeline ? "▾" : "▸"}</span>
              <span>Historical Backtesting &amp; Projections</span>
            </button>
            {showTimeline && (
              <HistoryOverlay
                simHistory={history}
                currentCoercion={config.coercion}
                currentMutationRate={config.mutationRate}
                width={timelineWidth}
                height={200}
              />
            )}
          </div>

          {/* Theology Engine */}
          <div className="w-full flex flex-col items-center">
            <button
              onClick={() => setShowTheology((v) => !v)}
              className="text-[10px] text-white/30 hover:text-white/50 transition-colors mb-1 flex items-center gap-1"
            >
              <span>{showTheology ? "▾" : "▸"}</span>
              <span>Theology Engine</span>
            </button>
            {showTheology && (
              <TheologyPanel
                snapshot={snapshot}
                width={timelineWidth}
              />
            )}
          </div>
        </main>

        {/* Right sidebar: Metrics */}
        <aside className="w-[260px] border-l border-white/5 p-4 overflow-y-auto flex-shrink-0">
          <MetricsPanel snapshot={snapshot} history={history} />
        </aside>
      </div>
    </div>
  );
}
