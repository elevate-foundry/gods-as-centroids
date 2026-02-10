"use client";

import { useRef, useEffect, useCallback } from "react";
import type { Snapshot } from "../lib/simulation";

// Cluster color palette — visually distinct, accessible
const PALETTE = [
  "#6366f1", // indigo
  "#f59e0b", // amber
  "#10b981", // emerald
  "#ef4444", // red
  "#8b5cf6", // violet
  "#06b6d4", // cyan
  "#f97316", // orange
  "#ec4899", // pink
  "#14b8a6", // teal
  "#a855f7", // purple
  "#eab308", // yellow
  "#3b82f6", // blue
  "#22c55e", // green
  "#e11d48", // rose
];

interface Props {
  snapshot: Snapshot | null;
  width: number;
  height: number;
}

export default function SimCanvas({ snapshot, width, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  // Smoothed agent positions for animation
  const smoothPos = useRef<Map<number, { x: number; y: number }>>(new Map());

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !snapshot) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = "#0a0a1a";
    ctx.fillRect(0, 0, width, height);

    // Draw subtle grid
    ctx.strokeStyle = "rgba(255,255,255,0.03)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Data-driven mapping: compute bounding box of all agent positions, then center + scale
    const margin = 70;
    const allX = snapshot.agents.map(a => a.x);
    const allY = snapshot.agents.map(a => a.y);
    const minX = Math.min(...allX, ...snapshot.clusters.map(c => c.cx));
    const maxX = Math.max(...allX, ...snapshot.clusters.map(c => c.cx));
    const minY = Math.min(...allY, ...snapshot.clusters.map(c => c.cy));
    const maxY = Math.max(...allY, ...snapshot.clusters.map(c => c.cy));
    const rangeX = (maxX - minX) || 0.1;
    const rangeY = (maxY - minY) || 0.1;
    // Add 15% padding around the data range
    const padX = rangeX * 0.15;
    const padY = rangeY * 0.15;
    const mapX = (v: number) => margin + ((v - minX + padX) / (rangeX + 2 * padX)) * (width - 2 * margin);
    const mapY = (v: number) => margin + ((v - minY + padY) / (rangeY + 2 * padY)) * (height - 2 * margin);

    // Smooth positions (lerp)
    const lerpFactor = 0.15;
    for (const agent of snapshot.agents) {
      const prev = smoothPos.current.get(agent.id);
      const tx = mapX(agent.x);
      const ty = mapY(agent.y);
      if (prev) {
        prev.x += (tx - prev.x) * lerpFactor;
        prev.y += (ty - prev.y) * lerpFactor;
      } else {
        smoothPos.current.set(agent.id, { x: tx, y: ty });
      }
    }

    // Draw attractor basins (soft radial gradients behind clusters)
    for (const cluster of snapshot.clusters) {
      if (cluster.agentIds.length < 2) continue;
      const cx = mapX(cluster.cx);
      const cy = mapY(cluster.cy);
      const radius = Math.sqrt(cluster.agentIds.length) * 25 + 30;
      const color = PALETTE[snapshot.clusters.indexOf(cluster) % PALETTE.length];

      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
      grad.addColorStop(0, color + "18");
      grad.addColorStop(0.6, color + "08");
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw agents
    for (const agent of snapshot.agents) {
      const pos = smoothPos.current.get(agent.id);
      if (!pos) continue;

      const color = PALETTE[agent.clusterId % PALETTE.length];
      const size = 3 + Math.min(agent.prestige * 0.8, 4);

      // Glow
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.fillStyle = color + "cc";
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, size, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Draw centroids (godforms) as stars
    for (let ci = 0; ci < snapshot.clusters.length; ci++) {
      const cluster = snapshot.clusters[ci];
      if (cluster.agentIds.length < 2) continue;

      const cx = mapX(cluster.cx);
      const cy = mapY(cluster.cy);
      const color = PALETTE[ci % PALETTE.length];
      const size = 8 + Math.sqrt(cluster.agentIds.length) * 2;

      // Outer glow
      ctx.shadowColor = color;
      ctx.shadowBlur = 20;

      // Draw star shape
      ctx.fillStyle = color;
      ctx.beginPath();
      const spikes = 6;
      for (let i = 0; i < spikes * 2; i++) {
        const r = i % 2 === 0 ? size : size * 0.4;
        const angle = (i * Math.PI) / spikes - Math.PI / 2;
        const sx = cx + Math.cos(angle) * r;
        const sy = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.closePath();
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 11px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(cluster.label, cx, cy + size + 14);

      // Follower count
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(`${cluster.agentIds.length} followers`, cx, cy + size + 26);
    }

    // Axis labels
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText("← Nurture · Care · Nature", width * 0.22, height - 12);
    ctx.fillText("Dominion · Authority · Power →", width * 0.78, height - 12);
    ctx.save();
    ctx.translate(14, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("← Chthonic · Death          Transcendence · Wisdom →", 0, 0);
    ctx.restore();

  }, [snapshot, width, height]);

  useEffect(() => {
    const loop = () => {
      draw();
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      className="rounded-xl border border-white/10"
      style={{ width, height }}
    />
  );
}
