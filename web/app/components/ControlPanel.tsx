"use client";

import { SimConfig, PRESETS } from "../lib/simulation";

interface Props {
  config: SimConfig;
  onChange: (partial: Partial<SimConfig>) => void;
  onReset: () => void;
  onPreset: (preset: Partial<SimConfig>) => void;
  running: boolean;
  onToggleRun: () => void;
  speed: number;
  onSpeedChange: (speed: number) => void;
  onProphet: () => void;
  onShock: (type: "war" | "plague" | "abundance" | "contact") => void;
  onSyncretism: () => void;
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  hint?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-baseline">
        <label className="text-xs font-medium text-white/70">{label}</label>
        <span className="text-xs font-mono text-white/50">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
                   [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-indigo-400 [&::-webkit-slider-thumb]:shadow-lg
                   [&::-webkit-slider-thumb]:shadow-indigo-500/30 [&::-webkit-slider-thumb]:cursor-pointer
                   hover:[&::-webkit-slider-thumb]:bg-indigo-300 transition-all"
      />
      {hint && <p className="text-[10px] text-white/30 leading-tight">{hint}</p>}
    </div>
  );
}

export default function ControlPanel({
  config,
  onChange,
  onReset,
  onPreset,
  running,
  onToggleRun,
  speed,
  onSpeedChange,
  onProphet,
  onShock,
  onSyncretism,
}: Props) {
  return (
    <div className="w-full h-full flex flex-col gap-4 overflow-y-auto pr-1 custom-scrollbar">
      {/* Playback controls */}
      <div className="flex gap-2">
        <button
          onClick={onToggleRun}
          className={`flex-1 px-4 py-2.5 rounded-lg font-semibold text-sm transition-all ${
            running
              ? "bg-red-500/20 text-red-300 border border-red-500/30 hover:bg-red-500/30"
              : "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-500/30"
          }`}
        >
          {running ? "⏸ Pause" : "▶ Run"}
        </button>
        <button
          onClick={onReset}
          className="px-4 py-2.5 rounded-lg text-sm font-medium text-white/60 border border-white/10
                     hover:bg-white/5 hover:text-white/80 transition-all"
        >
          ↻ Reset
        </button>
      </div>

      {/* Speed */}
      <Slider
        label="Simulation Speed"
        value={speed}
        min={1}
        max={20}
        step={1}
        onChange={onSpeedChange}
        hint="Steps per animation frame"
      />

      {/* Presets */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">
          Historical Scenarios
        </h3>
        <div className="grid gap-1.5">
          {PRESETS.map((preset) => (
            <button
              key={preset.name}
              onClick={() => onPreset(preset.config)}
              className="text-left px-3 py-2 rounded-lg border border-white/5 bg-white/[0.02]
                         hover:bg-white/[0.06] hover:border-white/10 transition-all group"
            >
              <div className="text-xs font-medium text-white/80 group-hover:text-white">
                {preset.name}
              </div>
              <div className="text-[10px] text-white/30 leading-tight mt-0.5">
                {preset.description}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-white/5" />

      {/* Historical Events */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">
          Inject Events
        </h3>
        <div className="grid grid-cols-2 gap-1.5">
          <button
            onClick={onProphet}
            className="px-2.5 py-2 rounded-lg border border-amber-500/20 bg-amber-500/5
                       hover:bg-amber-500/15 text-amber-300/80 hover:text-amber-200 text-xs font-medium transition-all"
          >
            ✦ Prophet
          </button>
          <button
            onClick={onSyncretism}
            className="px-2.5 py-2 rounded-lg border border-purple-500/20 bg-purple-500/5
                       hover:bg-purple-500/15 text-purple-300/80 hover:text-purple-200 text-xs font-medium transition-all"
          >
            ⊕ Syncretism
          </button>
          <button
            onClick={() => onShock("war")}
            className="px-2.5 py-2 rounded-lg border border-red-500/20 bg-red-500/5
                       hover:bg-red-500/15 text-red-300/80 hover:text-red-200 text-xs font-medium transition-all"
          >
            ⚔ War
          </button>
          <button
            onClick={() => onShock("plague")}
            className="px-2.5 py-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5
                       hover:bg-emerald-500/15 text-emerald-300/80 hover:text-emerald-200 text-xs font-medium transition-all"
          >
            ☠ Plague
          </button>
          <button
            onClick={() => onShock("abundance")}
            className="px-2.5 py-2 rounded-lg border border-cyan-500/20 bg-cyan-500/5
                       hover:bg-cyan-500/15 text-cyan-300/80 hover:text-cyan-200 text-xs font-medium transition-all"
          >
            ✿ Abundance
          </button>
          <button
            onClick={() => onShock("contact")}
            className="px-2.5 py-2 rounded-lg border border-blue-500/20 bg-blue-500/5
                       hover:bg-blue-500/15 text-blue-300/80 hover:text-blue-200 text-xs font-medium transition-all"
          >
            ⇄ Contact
          </button>
        </div>
        <p className="text-[10px] text-white/25 leading-tight">
          Inject historical events into the simulation. Prophet creates a charismatic leader.
          Syncretism merges the two closest traditions. Shocks perturb all beliefs.
        </p>
      </div>

      {/* Divider */}
      <div className="border-t border-white/5" />

      {/* Parameter sliders */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">
          Parameters
        </h3>

        <Slider
          label="Coercion"
          value={config.coercion}
          min={0}
          max={1}
          step={0.01}
          onChange={(v) => onChange({ coercion: v })}
          hint="Socio-political pressure toward religious homogeneity. High = monotheistic pressure."
        />

        <Slider
          label="Belief Influence"
          value={config.beliefInfluence}
          min={0}
          max={0.5}
          step={0.01}
          onChange={(v) => onChange({ beliefInfluence: v })}
          hint="How much an agent's existing beliefs bias their interpretation of new information."
        />

        <Slider
          label="Mutation Rate"
          value={config.mutationRate}
          min={0}
          max={0.3}
          step={0.01}
          onChange={(v) => onChange({ mutationRate: v })}
          hint="Doctrinal drift — random perturbations in belief. High = rapid theological innovation."
        />

        <Slider
          label="Prestige Weight"
          value={config.prestigeAlpha}
          min={0}
          max={0.8}
          step={0.01}
          onChange={(v) => onChange({ prestigeAlpha: v })}
          hint="How much successful communicators gain influence. High = charismatic leader effects."
        />

        <Slider
          label="Ritual Bonus"
          value={config.ritualBonus}
          min={0}
          max={0.4}
          step={0.01}
          onChange={(v) => onChange({ ritualBonus: v })}
          hint="Periodic ritual events boost communication success, stabilizing traditions."
        />

        <Slider
          label="Ritual Period"
          value={config.ritualPeriod}
          min={5}
          max={100}
          step={5}
          onChange={(v) => onChange({ ritualPeriod: v })}
          hint="Steps between ritual events. Lower = more frequent rituals."
        />

        <Slider
          label="Cluster Threshold"
          value={config.clusterThreshold}
          min={0.1}
          max={0.8}
          step={0.01}
          onChange={(v) => onChange({ clusterThreshold: v })}
          hint="Cosine distance threshold for forming new clusters. Lower = more distinct traditions."
        />

        <Slider
          label="Social Rewiring"
          value={config.socialP}
          min={0}
          max={0.5}
          step={0.01}
          onChange={(v) => onChange({ socialP: v })}
          hint="Watts-Strogatz rewiring probability. High = more long-range connections (trade routes, migration)."
        />
      </div>

      {/* Population */}
      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">
          Population (requires reset)
        </h3>
        <Slider
          label="Agents"
          value={config.N}
          min={20}
          max={200}
          step={10}
          onChange={(v) => onChange({ N: v })}
          hint="Number of belief-carrying agents in the swarm."
        />
      </div>
    </div>
  );
}
