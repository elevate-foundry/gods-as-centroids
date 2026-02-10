"use client";

import { useState, useCallback, useMemo } from "react";
import type { Snapshot } from "../lib/simulation";
import { AXES } from "../lib/simulation";
import {
  extractDeities,
  fuseDeities,
  interpolateDeities,
  type DeityObject,
} from "../lib/theology-engine";
import {
  encodeToBraille,
  latticeToUnicode,
  latticeToLabeledString,
  hammingDistance,
  testChannelInvariance,
} from "../lib/braille-lattice";

interface TheologyPanelProps {
  snapshot: Snapshot | null;
  width: number;
}

type TabId = "inspect" | "fuse" | "generate";
type GenAction = "describe" | "myth" | "prayer" | "doctrine";

const PALETTE = [
  "#6366f1", "#f59e0b", "#10b981", "#ef4444",
  "#8b5cf6", "#06b6d4", "#f97316", "#ec4899",
];

export default function TheologyPanel({ snapshot, width }: TheologyPanelProps) {
  const [activeTab, setActiveTab] = useState<TabId>("inspect");
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [selectedIdxB, setSelectedIdxB] = useState<number>(1);
  const [fuseWeights, setFuseWeights] = useState<number[]>([]);
  const [fuseSelection, setFuseSelection] = useState<Set<number>>(new Set());
  const [genAction, setGenAction] = useState<GenAction>("describe");
  const [genText, setGenText] = useState<string>("");
  const [genLoading, setGenLoading] = useState(false);
  const [genSource, setGenSource] = useState<string>("");
  const [interpT, setInterpT] = useState(0.5);
  const [showBrailleDetail, setShowBrailleDetail] = useState(false);

  const deities = useMemo(() => {
    if (!snapshot) return [];
    return extractDeities(snapshot, 2);
  }, [snapshot]);

  const selectedDeity = deities[selectedIdx] ?? null;
  const selectedDeityB = deities[selectedIdxB] ?? null;

  const fusedDeity = useMemo(() => {
    if (fuseSelection.size < 2) return null;
    const selected = Array.from(fuseSelection).map((i) => deities[i]).filter(Boolean);
    if (selected.length < 2) return null;
    const w = fuseWeights.length === selected.length ? fuseWeights : selected.map(() => 1 / selected.length);
    return fuseDeities(selected, w);
  }, [deities, fuseSelection, fuseWeights]);

  const interpolated = useMemo(() => {
    if (!selectedDeity || !selectedDeityB) return null;
    return interpolateDeities(selectedDeity, selectedDeityB, interpT);
  }, [selectedDeity, selectedDeityB, interpT]);

  const brailleSignature = useCallback((deity: DeityObject) => {
    const lattice = encodeToBraille(deity.vector);
    return latticeToUnicode(lattice);
  }, []);

  const brailleDetailed = useCallback((deity: DeityObject) => {
    const lattice = encodeToBraille(deity.vector);
    return latticeToLabeledString(lattice);
  }, []);

  const handleGenerate = useCallback(async (deity: DeityObject, action: GenAction) => {
    setGenLoading(true);
    setGenText("");
    setGenSource("");
    try {
      const brailleSig = brailleSignature(deity);
      const res = await fetch("/api/theology", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          deity: {
            name: deity.name,
            vector: deity.vector,
            dominantAxes: deity.dominantAxes,
            recessiveAxes: deity.recessiveAxes,
            source: deity.source,
            adherents: deity.adherents,
            profile: deity.profile,
            brailleSignature: brailleSig,
            parentNames: deity.parentIds,
            mixingWeights: deity.mixingWeights,
          },
        }),
      });
      const data = await res.json();
      setGenText(data.text || data.error || "No response");
      setGenSource(data.source || "unknown");
    } catch (e) {
      setGenText(`Error: ${e}`);
    } finally {
      setGenLoading(false);
    }
  }, [brailleSignature]);

  const handleFuseGenerate = useCallback(async () => {
    if (!fusedDeity) return;
    setGenLoading(true);
    setGenText("");
    try {
      const parentNames = Array.from(fuseSelection).map((i) => deities[i]?.name).filter(Boolean);
      const brailleSig = brailleSignature(fusedDeity);
      const res = await fetch("/api/theology", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "fuse_describe",
          deity: {
            name: fusedDeity.name,
            vector: fusedDeity.vector,
            dominantAxes: fusedDeity.dominantAxes,
            recessiveAxes: fusedDeity.recessiveAxes,
            source: fusedDeity.source,
            adherents: fusedDeity.adherents,
            profile: fusedDeity.profile,
            brailleSignature: brailleSig,
            parentNames,
            mixingWeights: fusedDeity.mixingWeights,
          },
        }),
      });
      const data = await res.json();
      setGenText(data.text || data.error || "No response");
      setGenSource(data.source || "unknown");
    } catch (e) {
      setGenText(`Error: ${e}`);
    } finally {
      setGenLoading(false);
    }
  }, [fusedDeity, fuseSelection, deities, brailleSignature]);

  if (!snapshot || deities.length === 0) {
    return (
      <div className="p-4 text-white/30 text-xs text-center">
        Run the simulation to see emergent deities...
      </div>
    );
  }

  return (
    <div style={{ width }} className="bg-[#0d0d12] border border-white/5 rounded-lg overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-white/5">
        {(["inspect", "fuse", "generate"] as TabId[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 px-3 py-2 text-[11px] font-medium uppercase tracking-wider transition-colors ${
              activeTab === tab
                ? "text-indigo-400 border-b-2 border-indigo-400 bg-white/[0.02]"
                : "text-white/30 hover:text-white/50"
            }`}
          >
            {tab === "inspect" ? "⬡ Inspect" : tab === "fuse" ? "⊕ Fuse" : "✦ Generate"}
          </button>
        ))}
      </div>

      <div className="p-3 max-h-[500px] overflow-y-auto">
        {/* ─── INSPECT TAB ─── */}
        {activeTab === "inspect" && selectedDeity && (
          <div className="space-y-3">
            {/* Deity selector */}
            <div className="flex gap-1.5 flex-wrap">
              {deities.map((d, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedIdx(i)}
                  className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                    i === selectedIdx
                      ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                      : "bg-white/[0.03] text-white/50 border border-white/5 hover:border-white/15"
                  }`}
                >
                  <span
                    className="inline-block w-2 h-2 rounded-full mr-1.5"
                    style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
                  />
                  {d.name}
                </button>
              ))}
            </div>

            {/* Deity header */}
            <div className="bg-white/[0.02] rounded-lg p-3 border border-white/5">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-white font-semibold text-sm">{selectedDeity.name}</h3>
                  <p className="text-indigo-400 text-[11px] font-medium">{selectedDeity.profile.archetype}</p>
                </div>
                <div className="text-right">
                  <div className="text-white/40 text-[10px]">{selectedDeity.adherents} adherents</div>
                  <div className="text-white/30 text-[10px]">{selectedDeity.source}</div>
                </div>
              </div>

              {/* Braille signature */}
              <div className="mt-3 bg-black/30 rounded p-2">
                <div className="text-[10px] text-white/30 uppercase tracking-wider mb-1">Braille Signature (72-bit)</div>
                <div className="text-2xl tracking-[0.3em] text-indigo-300 font-mono">
                  {brailleSignature(selectedDeity)}
                </div>
                <button
                  onClick={() => setShowBrailleDetail(!showBrailleDetail)}
                  className="text-[9px] text-white/20 hover:text-white/40 mt-1"
                >
                  {showBrailleDetail ? "▾ hide detail" : "▸ show per-axis"}
                </button>
                {showBrailleDetail && (
                  <div className="mt-1 text-[10px] text-white/30 font-mono whitespace-pre-wrap leading-relaxed">
                    {brailleDetailed(selectedDeity)}
                  </div>
                )}
              </div>
            </div>

            {/* Vector heatmap */}
            <div className="bg-white/[0.02] rounded-lg p-3 border border-white/5">
              <div className="text-[10px] text-white/30 uppercase tracking-wider mb-2">Theological Vector</div>
              <div className="space-y-1">
                {AXES.map((axis) => {
                  const val = selectedDeity.vector[axis];
                  const isDominant = selectedDeity.dominantAxes.includes(axis);
                  const isRecessive = selectedDeity.recessiveAxes.includes(axis);
                  return (
                    <div key={axis} className="flex items-center gap-2">
                      <span className={`text-[10px] w-24 text-right font-mono ${
                        isDominant ? "text-indigo-400 font-bold" : isRecessive ? "text-red-400/50" : "text-white/40"
                      }`}>
                        {axis}
                      </span>
                      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${Math.abs(val) * 100}%`,
                            backgroundColor: isDominant ? "#6366f1" : isRecessive ? "#ef4444" : "#ffffff20",
                          }}
                        />
                      </div>
                      <span className="text-[10px] text-white/30 font-mono w-10 text-right">
                        {val.toFixed(3)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Profile */}
            <div className="bg-white/[0.02] rounded-lg p-3 border border-white/5">
              <div className="text-[10px] text-white/30 uppercase tracking-wider mb-2">Theological Profile</div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
                {Object.entries(selectedDeity.profile).map(([key, val]) => (
                  <div key={key}>
                    <span className="text-white/30">{key.replace(/([A-Z])/g, " $1").trim()}: </span>
                    <span className="text-white/70">{Array.isArray(val) ? val.join(", ") : val}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick generate buttons */}
            <div className="flex gap-1.5">
              {(["describe", "myth", "prayer", "doctrine"] as GenAction[]).map((action) => (
                <button
                  key={action}
                  onClick={() => {
                    setGenAction(action);
                    handleGenerate(selectedDeity, action);
                  }}
                  disabled={genLoading}
                  className="flex-1 px-2 py-1.5 bg-indigo-500/10 border border-indigo-500/20 rounded text-[10px] text-indigo-300 hover:bg-indigo-500/20 transition-colors disabled:opacity-30"
                >
                  {action === "describe" ? "📜 Describe" : action === "myth" ? "🌅 Myth" : action === "prayer" ? "🙏 Prayer" : "📋 Doctrine"}
                </button>
              ))}
            </div>

            {/* Generated text */}
            {(genText || genLoading) && (
              <div className="bg-black/30 rounded-lg p-3 border border-white/5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] text-white/30 uppercase tracking-wider">
                    {genLoading ? "Generating..." : `Generated ${genAction}`}
                  </span>
                  {genSource && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-white/20">
                      {genSource}
                    </span>
                  )}
                </div>
                {genLoading ? (
                  <div className="flex items-center gap-2 text-white/30 text-xs">
                    <div className="w-3 h-3 border-2 border-indigo-400/30 border-t-indigo-400 rounded-full animate-spin" />
                    Consulting the theology engine...
                  </div>
                ) : (
                  <div className="text-white/70 text-[12px] leading-relaxed whitespace-pre-wrap">
                    {genText}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ─── FUSE TAB ─── */}
        {activeTab === "fuse" && (
          <div className="space-y-3">
            <div className="text-[10px] text-white/30 uppercase tracking-wider">
              Select 2+ deities to fuse into a meta-centroid
            </div>

            {/* Selection grid */}
            <div className="flex gap-1.5 flex-wrap">
              {deities.map((d, i) => {
                const selected = fuseSelection.has(i);
                return (
                  <button
                    key={i}
                    onClick={() => {
                      const next = new Set(fuseSelection);
                      if (selected) next.delete(i);
                      else next.add(i);
                      setFuseSelection(next);
                      setFuseWeights([]);
                    }}
                    className={`px-2.5 py-1.5 rounded text-[11px] font-medium transition-all ${
                      selected
                        ? "bg-green-500/20 text-green-300 border border-green-500/40"
                        : "bg-white/[0.03] text-white/50 border border-white/5 hover:border-white/15"
                    }`}
                  >
                    <span
                      className="inline-block w-2 h-2 rounded-full mr-1.5"
                      style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
                    />
                    {d.name}
                    {selected && " ✓"}
                  </button>
                );
              })}
            </div>

            {/* Weight sliders */}
            {fuseSelection.size >= 2 && (
              <div className="bg-white/[0.02] rounded-lg p-3 border border-white/5 space-y-2">
                <div className="text-[10px] text-white/30 uppercase tracking-wider">Mixing Weights (α)</div>
                {Array.from(fuseSelection).map((idx, j) => {
                  const w = fuseWeights[j] ?? 1 / fuseSelection.size;
                  return (
                    <div key={idx} className="flex items-center gap-2">
                      <span className="text-[11px] text-white/50 w-16 truncate">{deities[idx]?.name}</span>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={w * 100}
                        onChange={(e) => {
                          const newWeights = [...(fuseWeights.length ? fuseWeights : Array.from(fuseSelection).map(() => 1 / fuseSelection.size))];
                          newWeights[j] = parseInt(e.target.value) / 100;
                          setFuseWeights(newWeights);
                        }}
                        className="flex-1 h-1 accent-green-500"
                      />
                      <span className="text-[10px] text-white/30 font-mono w-8">{(w * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Fused result */}
            {fusedDeity && (
              <div className="bg-green-500/[0.04] rounded-lg p-3 border border-green-500/10 space-y-2">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-green-300 font-semibold text-sm">{fusedDeity.name}</h3>
                    <p className="text-green-400/60 text-[11px]">{fusedDeity.profile.archetype}</p>
                  </div>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-green-500/10 text-green-400/50">
                    meta-centroid
                  </span>
                </div>

                {/* Fused braille */}
                <div className="bg-black/30 rounded p-2">
                  <div className="text-[10px] text-white/30 mb-1">Braille Signature</div>
                  <div className="text-xl tracking-[0.3em] text-green-300 font-mono">
                    {brailleSignature(fusedDeity)}
                  </div>
                </div>

                {/* Fused vector bars */}
                <div className="space-y-0.5">
                  {AXES.map((axis) => {
                    const val = fusedDeity.vector[axis];
                    const isDominant = fusedDeity.dominantAxes.includes(axis);
                    return (
                      <div key={axis} className="flex items-center gap-1.5">
                        <span className={`text-[9px] w-20 text-right font-mono ${isDominant ? "text-green-400" : "text-white/30"}`}>
                          {axis}
                        </span>
                        <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.abs(val) * 100}%`,
                              backgroundColor: isDominant ? "#10b981" : "#ffffff15",
                            }}
                          />
                        </div>
                        <span className="text-[9px] text-white/20 font-mono w-8">{val.toFixed(3)}</span>
                      </div>
                    );
                  })}
                </div>

                <button
                  onClick={handleFuseGenerate}
                  disabled={genLoading}
                  className="w-full px-3 py-2 bg-green-500/10 border border-green-500/20 rounded text-[11px] text-green-300 hover:bg-green-500/20 transition-colors disabled:opacity-30"
                >
                  {genLoading ? "Generating..." : "✦ Generate Theology for Fused Deity"}
                </button>
              </div>
            )}

            {/* Interpolation */}
            {deities.length >= 2 && (
              <div className="bg-white/[0.02] rounded-lg p-3 border border-white/5 space-y-2">
                <div className="text-[10px] text-white/30 uppercase tracking-wider">Interpolation (Slerp)</div>
                <div className="flex gap-2 items-center">
                  <select
                    value={selectedIdx}
                    onChange={(e) => setSelectedIdx(parseInt(e.target.value))}
                    className="bg-black/30 border border-white/10 rounded px-2 py-1 text-[11px] text-white/60"
                  >
                    {deities.map((d, i) => (
                      <option key={i} value={i}>{d.name}</option>
                    ))}
                  </select>
                  <span className="text-white/20 text-[10px]">↔</span>
                  <select
                    value={selectedIdxB}
                    onChange={(e) => setSelectedIdxB(parseInt(e.target.value))}
                    className="bg-black/30 border border-white/10 rounded px-2 py-1 text-[11px] text-white/60"
                  >
                    {deities.map((d, i) => (
                      <option key={i} value={i}>{d.name}</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-white/30">t=0</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={interpT * 100}
                    onChange={(e) => setInterpT(parseInt(e.target.value) / 100)}
                    className="flex-1 h-1 accent-cyan-500"
                  />
                  <span className="text-[10px] text-white/30">t=1</span>
                  <span className="text-[10px] text-cyan-400 font-mono w-8">{interpT.toFixed(2)}</span>
                </div>
                {interpolated && (
                  <div className="bg-black/20 rounded p-2">
                    <div className="text-cyan-300 text-[11px] font-medium">{interpolated.profile.archetype}</div>
                    <div className="text-lg tracking-[0.2em] text-cyan-300/70 font-mono mt-1">
                      {brailleSignature(interpolated)}
                    </div>
                    <div className="text-[10px] text-white/30 mt-1">
                      Dominant: {interpolated.dominantAxes.join(", ")}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Generated text for fused deity */}
            {(genText || genLoading) && activeTab === "fuse" && (
              <div className="bg-black/30 rounded-lg p-3 border border-white/5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] text-white/30 uppercase tracking-wider">
                    {genLoading ? "Generating..." : "Fused Theology"}
                  </span>
                  {genSource && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-white/20">{genSource}</span>
                  )}
                </div>
                {genLoading ? (
                  <div className="flex items-center gap-2 text-white/30 text-xs">
                    <div className="w-3 h-3 border-2 border-green-400/30 border-t-green-400 rounded-full animate-spin" />
                    Synthesizing meta-theology...
                  </div>
                ) : (
                  <div className="text-white/70 text-[12px] leading-relaxed whitespace-pre-wrap">{genText}</div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ─── GENERATE TAB ─── */}
        {activeTab === "generate" && (
          <div className="space-y-3">
            <div className="text-[10px] text-white/30 uppercase tracking-wider">
              Compare two deities
            </div>

            <div className="flex gap-2 items-center">
              <select
                value={selectedIdx}
                onChange={(e) => setSelectedIdx(parseInt(e.target.value))}
                className="flex-1 bg-black/30 border border-white/10 rounded px-2 py-1.5 text-[11px] text-white/60"
              >
                {deities.map((d, i) => (
                  <option key={i} value={i}>{d.name}</option>
                ))}
              </select>
              <span className="text-white/20 text-xs">vs</span>
              <select
                value={selectedIdxB}
                onChange={(e) => setSelectedIdxB(parseInt(e.target.value))}
                className="flex-1 bg-black/30 border border-white/10 rounded px-2 py-1.5 text-[11px] text-white/60"
              >
                {deities.map((d, i) => (
                  <option key={i} value={i}>{d.name}</option>
                ))}
              </select>
            </div>

            {/* Side-by-side braille */}
            {selectedDeity && selectedDeityB && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-white/[0.02] rounded p-2 border border-white/5">
                    <div className="text-white/70 text-[11px] font-medium">{selectedDeity.name}</div>
                    <div className="text-lg tracking-[0.2em] text-indigo-300 font-mono mt-1">
                      {brailleSignature(selectedDeity)}
                    </div>
                    <div className="text-[10px] text-white/30 mt-1">{selectedDeity.profile.archetype}</div>
                  </div>
                  <div className="bg-white/[0.02] rounded p-2 border border-white/5">
                    <div className="text-white/70 text-[11px] font-medium">{selectedDeityB.name}</div>
                    <div className="text-lg tracking-[0.2em] text-amber-300 font-mono mt-1">
                      {brailleSignature(selectedDeityB)}
                    </div>
                    <div className="text-[10px] text-white/30 mt-1">{selectedDeityB.profile.archetype}</div>
                  </div>
                </div>

                {/* Hamming distance */}
                <div className="bg-white/[0.02] rounded p-2 border border-white/5 text-center">
                  <span className="text-[10px] text-white/30">Hamming Distance: </span>
                  <span className="text-cyan-400 font-mono text-sm font-bold">
                    {hammingDistance(encodeToBraille(selectedDeity.vector), encodeToBraille(selectedDeityB.vector))}
                  </span>
                  <span className="text-[10px] text-white/30"> / 72 bits</span>
                </div>

                <button
                  onClick={async () => {
                    setGenLoading(true);
                    setGenText("");
                    try {
                      const res = await fetch("/api/theology", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          action: "compare",
                          deity: {
                            name: selectedDeity.name,
                            vector: selectedDeity.vector,
                            dominantAxes: selectedDeity.dominantAxes,
                            recessiveAxes: selectedDeity.recessiveAxes,
                            source: selectedDeity.source,
                            adherents: selectedDeity.adherents,
                            profile: selectedDeity.profile,
                          },
                          deityB: {
                            name: selectedDeityB.name,
                            vector: selectedDeityB.vector,
                            dominantAxes: selectedDeityB.dominantAxes,
                            profile: selectedDeityB.profile,
                          },
                        }),
                      });
                      const data = await res.json();
                      setGenText(data.text || data.error);
                      setGenSource(data.source || "unknown");
                    } catch (e) {
                      setGenText(`Error: ${e}`);
                    } finally {
                      setGenLoading(false);
                    }
                  }}
                  disabled={genLoading}
                  className="w-full px-3 py-2 bg-cyan-500/10 border border-cyan-500/20 rounded text-[11px] text-cyan-300 hover:bg-cyan-500/20 transition-colors disabled:opacity-30"
                >
                  {genLoading ? "Comparing..." : "⚡ Compare Deities"}
                </button>
              </>
            )}

            {/* Generated comparison */}
            {(genText || genLoading) && activeTab === "generate" && (
              <div className="bg-black/30 rounded-lg p-3 border border-white/5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] text-white/30 uppercase tracking-wider">
                    {genLoading ? "Comparing..." : "Comparative Theology"}
                  </span>
                  {genSource && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-white/20">{genSource}</span>
                  )}
                </div>
                {genLoading ? (
                  <div className="flex items-center gap-2 text-white/30 text-xs">
                    <div className="w-3 h-3 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
                    Analyzing theological divergence...
                  </div>
                ) : (
                  <div className="text-white/70 text-[12px] leading-relaxed whitespace-pre-wrap">{genText}</div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
