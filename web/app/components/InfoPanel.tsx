"use client";

import { useState } from "react";

const SECTIONS = [
  {
    title: "What is this?",
    content: `This is an interactive simulation of the "Gods as Centroids" model — a computational theory of religious evolution. Each dot is a believer carrying a 12-dimensional belief vector. As they interact on a social network, their beliefs cluster. The center of each cluster — the centroid — is a mathematically emergent "godform."`,
  },
  {
    title: "The 12 Dimensions",
    content: `Every agent's belief is a vector across: Authority, Transcendence, Care, Justice, Wisdom, Power, Fertility, War, Death, Creation, Nature, and Order. These axes are grounded in Cognitive Science of Religion — they represent the core dimensions along which human religious thought varies.`,
  },
  {
    title: "Coercion & Phase Transitions",
    content: `The coercion parameter models socio-political pressure toward religious homogeneity (state religion, doctrinal exclusivity). At low coercion, multiple godforms coexist (polytheism). As coercion increases past a critical threshold, the system undergoes a phase transition — one centroid swallows all others (monotheism). This transition exhibits hysteresis: reducing coercion doesn't easily restore pluralism.`,
  },
  {
    title: "Key Predictions",
    content: `A) Universality: Any swarm with sufficient social coupling produces godforms. B) Monotheism transition: N_eff decreases monotonically with coercion. C) Syncretism: Cluster merging at cultural crossroads. D) Ritual stabilization: Higher ritual costs reduce tradition churn. E) Prestige amplification: Charismatic leaders accelerate convergence.`,
  },
  {
    title: "How to Explore",
    content: `Try the presets to see historical scenarios in action. Then experiment: crank coercion to 0.9 and watch monotheism emerge. Drop it back — notice the hysteresis. Raise mutation rate to simulate a Reformation. Max out prestige to see a charismatic prophet reshape the landscape. Each parameter maps to a real sociological force.`,
  },
];

export default function InfoPanel() {
  const [openIdx, setOpenIdx] = useState<number | null>(0);

  return (
    <div className="space-y-1">
      {SECTIONS.map((section, i) => (
        <div key={i}>
          <button
            onClick={() => setOpenIdx(openIdx === i ? null : i)}
            className="w-full text-left px-3 py-2 rounded-lg hover:bg-white/[0.03] transition-colors
                       flex items-center justify-between group"
          >
            <span className="text-xs font-medium text-white/60 group-hover:text-white/80">
              {section.title}
            </span>
            <span className="text-white/30 text-xs">
              {openIdx === i ? "−" : "+"}
            </span>
          </button>
          {openIdx === i && (
            <div className="px-3 pb-3 text-[11px] text-white/40 leading-relaxed">
              {section.content}
            </div>
          )}
        </div>
      ))}

      <div className="border-t border-white/5 pt-3 mt-3">
        <div className="px-3 text-[10px] text-white/20 leading-relaxed">
          <p className="font-medium text-white/30 mb-1">Citation</p>
          <p>
            Barrett, R. (2025). &ldquo;Gods as Centroids: A Swarm-Based Vector
            Model of Religious Evolution.&rdquo;
          </p>
          <p className="mt-2">
            Built on a 12-axis belief space with Watts-Strogatz social networks,
            prestige-weighted transmission, and online centroid clustering.
            Implementations in Python, Go, Rust, and C++.
          </p>
        </div>
      </div>
    </div>
  );
}
