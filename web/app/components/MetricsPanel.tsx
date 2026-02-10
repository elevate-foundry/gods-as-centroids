"use client";

import { Snapshot } from "../lib/simulation";

interface Props {
  snapshot: Snapshot | null;
  history: { t: number; nEff: number; entropy: number; dominance: number }[];
}

function Metric({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
}) {
  return (
    <div className="bg-white/[0.03] rounded-lg px-3 py-2.5 border border-white/5">
      <div className="text-[10px] text-white/40 uppercase tracking-wider font-medium">
        {label}
      </div>
      <div className={`text-xl font-bold font-mono mt-0.5 ${color || "text-white"}`}>
        {value}
      </div>
      {sub && <div className="text-[10px] text-white/30 mt-0.5">{sub}</div>}
    </div>
  );
}

function MiniChart({
  data,
  color,
  height = 40,
}: {
  data: number[];
  color: string;
  height?: number;
}) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const w = 200;

  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${w} ${height}`}
      className="w-full"
      style={{ height }}
      preserveAspectRatio="none"
    >
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.7"
      />
    </svg>
  );
}

export default function MetricsPanel({ snapshot, history }: Props) {
  if (!snapshot) return null;

  const phaseColors: Record<string, string> = {
    polytheistic: "text-emerald-400",
    transitional: "text-amber-400",
    monotheistic: "text-red-400",
  };

  const phaseEmoji: Record<string, string> = {
    polytheistic: "☀",
    transitional: "⚡",
    monotheistic: "✦",
  };

  return (
    <div className="space-y-3">
      {/* Phase indicator */}
      <div
        className={`text-center py-2 rounded-lg border ${
          snapshot.phase === "polytheistic"
            ? "bg-emerald-500/10 border-emerald-500/20"
            : snapshot.phase === "monotheistic"
            ? "bg-red-500/10 border-red-500/20"
            : "bg-amber-500/10 border-amber-500/20"
        }`}
      >
        <span className={`text-sm font-semibold ${phaseColors[snapshot.phase]}`}>
          {phaseEmoji[snapshot.phase]} {snapshot.phase.charAt(0).toUpperCase() + snapshot.phase.slice(1)} Phase
        </span>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 gap-2">
        <Metric
          label="Godforms (N_eff)"
          value={snapshot.nEff.toString()}
          sub="Active centroids"
          color={
            snapshot.nEff <= 2
              ? "text-red-400"
              : snapshot.nEff <= 4
              ? "text-amber-400"
              : "text-emerald-400"
          }
        />
        <Metric
          label="Dominance"
          value={`${(snapshot.dominance * 100).toFixed(0)}%`}
          sub="Largest tradition"
          color={snapshot.dominance > 0.6 ? "text-red-400" : "text-white"}
        />
        <Metric
          label="Entropy"
          value={snapshot.entropy.toFixed(2)}
          sub="Diversity measure"
          color="text-cyan-400"
        />
        <Metric
          label="Time Step"
          value={snapshot.t.toString()}
          sub={`${snapshot.agents.length} agents`}
          color="text-white/60"
        />
      </div>

      {/* Mini charts */}
      <div className="space-y-2">
        <div className="bg-white/[0.03] rounded-lg p-2 border border-white/5">
          <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1">
            Godforms over time
          </div>
          <MiniChart data={history.map((h) => h.nEff)} color="#6366f1" />
        </div>
        <div className="bg-white/[0.03] rounded-lg p-2 border border-white/5">
          <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1">
            Dominance over time
          </div>
          <MiniChart data={history.map((h) => h.dominance)} color="#ef4444" />
        </div>
        <div className="bg-white/[0.03] rounded-lg p-2 border border-white/5">
          <div className="text-[10px] text-white/40 uppercase tracking-wider mb-1">
            Entropy over time
          </div>
          <MiniChart data={history.map((h) => h.entropy)} color="#06b6d4" />
        </div>
      </div>

      {/* Cluster breakdown */}
      {snapshot.clusters.length > 0 && (
        <div className="space-y-1.5">
          <h4 className="text-[10px] text-white/40 uppercase tracking-wider font-medium">
            Active Traditions
          </h4>
          {snapshot.clusters
            .filter((c) => c.agentIds.length >= 2)
            .sort((a, b) => b.agentIds.length - a.agentIds.length)
            .map((cluster, i) => {
              const pct = (cluster.agentIds.length / snapshot.agents.length) * 100;
              return (
                <div
                  key={i}
                  className="flex items-center gap-2 text-xs"
                >
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{
                      backgroundColor: [
                        "#6366f1", "#f59e0b", "#10b981", "#ef4444",
                        "#8b5cf6", "#06b6d4", "#f97316", "#ec4899",
                      ][i % 8],
                    }}
                  />
                  <span className="text-white/70 flex-1 truncate">
                    {cluster.label}
                  </span>
                  <span className="text-white/40 font-mono text-[10px]">
                    {cluster.agentIds.length} ({pct.toFixed(0)}%)
                  </span>
                  <div className="w-16 h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: [
                          "#6366f1", "#f59e0b", "#10b981", "#ef4444",
                          "#8b5cf6", "#06b6d4", "#f97316", "#ec4899",
                        ][i % 8],
                      }}
                    />
                  </div>
                </div>
              );
            })}
        </div>
      )}
    </div>
  );
}
