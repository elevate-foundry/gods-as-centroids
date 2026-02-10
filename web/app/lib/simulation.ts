/**
 * Gods as Centroids: SwarmKernel — TypeScript port
 * A swarm-based vector model of religious evolution.
 *
 * Core simulation engine: agents carry belief vectors in a 12-dimensional
 * theological space. They interact on a small-world social network,
 * producing and interpreting symbolic forms. Clustering yields emergent
 * "godform" centroids — the mathematical deities.
 */

// ─── Axes ────────────────────────────────────────────────────────────
export const AXES = [
  "authority", "transcendence", "care", "justice", "wisdom", "power",
  "fertility", "war", "death", "creation", "nature", "order",
] as const;

export type Axis = (typeof AXES)[number];
export type Vec = Record<Axis, number>;

// ─── Deity Priors ────────────────────────────────────────────────────
export const DEITY_PRIORS: Record<string, Vec> = {
  zeus:    { authority:0.9, transcendence:0.8, care:0.3, justice:0.7, wisdom:0.6, power:0.9, fertility:0.2, war:0.8, death:0.1, creation:0.4, nature:0.3, order:0.8 },
  odin:    { authority:0.8, transcendence:0.7, care:0.4, justice:0.6, wisdom:0.9, power:0.7, fertility:0.1, war:0.9, death:0.8, creation:0.3, nature:0.2, order:0.5 },
  amun:    { authority:0.9, transcendence:0.9, care:0.6, justice:0.8, wisdom:0.8, power:0.8, fertility:0.3, war:0.2, death:0.1, creation:0.9, nature:0.1, order:0.9 },
  marduk:  { authority:0.9, transcendence:0.6, care:0.5, justice:0.9, wisdom:0.7, power:0.9, fertility:0.1, war:0.8, death:0.3, creation:0.7, nature:0.1, order:0.9 },
  indra:   { authority:0.8, transcendence:0.5, care:0.4, justice:0.7, wisdom:0.6, power:0.9, fertility:0.2, war:0.9, death:0.2, creation:0.3, nature:0.4, order:0.6 },
  shango:  { authority:0.7, transcendence:0.4, care:0.3, justice:0.8, wisdom:0.5, power:0.8, fertility:0.1, war:0.7, death:0.2, creation:0.2, nature:0.6, order:0.5 },
  kami:    { authority:0.3, transcendence:0.8, care:0.8, justice:0.4, wisdom:0.7, power:0.4, fertility:0.6, war:0.1, death:0.1, creation:0.5, nature:0.9, order:0.8 },
  manitou: { authority:0.2, transcendence:0.9, care:0.9, justice:0.3, wisdom:0.8, power:0.3, fertility:0.7, war:0.1, death:0.2, creation:0.6, nature:0.9, order:0.4 },
  apollo:  { authority:0.6, transcendence:0.7, care:0.5, justice:0.6, wisdom:0.8, power:0.6, fertility:0.3, war:0.3, death:0.2, creation:0.7, nature:0.4, order:0.7 },
  freya:   { authority:0.4, transcendence:0.5, care:0.8, justice:0.4, wisdom:0.6, power:0.5, fertility:0.9, war:0.6, death:0.4, creation:0.6, nature:0.7, order:0.3 },
  isis:    { authority:0.4, transcendence:0.6, care:0.9, justice:0.6, wisdom:0.8, power:0.5, fertility:0.8, war:0.1, death:0.4, creation:0.7, nature:0.5, order:0.6 },
  ra:      { authority:0.8, transcendence:0.8, care:0.5, justice:0.7, wisdom:0.7, power:0.8, fertility:0.3, war:0.4, death:0.2, creation:0.8, nature:0.6, order:0.8 },
  yah:     { authority:0.9, transcendence:0.9, care:0.6, justice:0.9, wisdom:0.9, power:0.9, fertility:0.2, war:0.4, death:0.3, creation:0.9, nature:0.3, order:0.9 },
  baal:    { authority:0.7, transcendence:0.6, care:0.4, justice:0.6, wisdom:0.5, power:0.8, fertility:0.6, war:0.6, death:0.3, creation:0.5, nature:0.7, order:0.6 },
};

// Normalize all priors to unit vectors
for (const name of Object.keys(DEITY_PRIORS)) {
  const v = DEITY_PRIORS[name];
  const n = Math.sqrt(AXES.reduce((s, a) => s + v[a] * v[a], 0)) || 1;
  for (const a of AXES) v[a] /= n;
}

// ─── Vector helpers ──────────────────────────────────────────────────
export function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (const k of AXES) s += a[k] * b[k];
  return s;
}
export function vecNorm(a: Vec): number {
  return Math.sqrt(dot(a, a));
}
export function cosine(a: Vec, b: Vec): number {
  const na = vecNorm(a), nb = vecNorm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}
export function zeroVec(): Vec {
  const v = {} as Vec;
  for (const a of AXES) v[a] = 0;
  return v;
}
export function addScaled(target: Vec, src: Vec, s: number) {
  for (const a of AXES) target[a] += src[a] * s;
}
export function scaleVec(v: Vec, s: number): Vec {
  const out = {} as Vec;
  for (const a of AXES) out[a] = v[a] * s;
  return out;
}

// ─── Seeded RNG (xoshiro128**) ───────────────────────────────────────
class RNG {
  private s: Uint32Array;
  constructor(seed: number) {
    this.s = new Uint32Array(4);
    this.s[0] = seed >>> 0;
    this.s[1] = (seed * 1812433253 + 1) >>> 0;
    this.s[2] = (this.s[1] * 1812433253 + 1) >>> 0;
    this.s[3] = (this.s[2] * 1812433253 + 1) >>> 0;
    for (let i = 0; i < 20; i++) this._next();
  }
  private _next(): number {
    const s = this.s;
    const result = Math.imul(s[1] * 5, 1 << 7 | 1 >>> 25) >>> 0;
    const t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = (s[3] << 11 | s[3] >>> 21) >>> 0;
    return result;
  }
  random(): number { return this._next() / 4294967296; }
  gauss(mu = 0, sigma = 1): number {
    let u, v, s2;
    do { u = 2 * this.random() - 1; v = 2 * this.random() - 1; s2 = u * u + v * v; } while (s2 >= 1 || s2 === 0);
    return mu + sigma * u * Math.sqrt(-2 * Math.log(s2) / s2);
  }
  randInt(lo: number, hi: number): number { return lo + Math.floor(this.random() * (hi - lo)); }
  choice<T>(arr: T[]): T { return arr[Math.floor(this.random() * arr.length)]; }
  sample<T>(arr: T[], k: number): T[] {
    const copy = [...arr];
    const out: T[] = [];
    for (let i = 0; i < Math.min(k, copy.length); i++) {
      const j = this.randInt(i, copy.length);
      [copy[i], copy[j]] = [copy[j], copy[i]];
      out.push(copy[i]);
    }
    return out;
  }
  weightedChoice<T>(items: T[], weights: number[]): T {
    const total = weights.reduce((a, b) => a + b, 0) || 1;
    let r = this.random() * total;
    for (let i = 0; i < items.length; i++) {
      r -= weights[i];
      if (r <= 0) return items[i];
    }
    return items[items.length - 1];
  }
}

function rndUnitVec(rng: RNG): Vec {
  const v = {} as Vec;
  for (const a of AXES) v[a] = rng.random();
  const n = vecNorm(v) || 1;
  for (const a of AXES) v[a] /= n;
  return v;
}

function jitterVec(base: Vec, rng: RNG, noise = 0.1): Vec {
  const v = {} as Vec;
  for (const a of AXES) v[a] = base[a] + rng.gauss(0, noise);
  const n = vecNorm(v) || 1;
  for (const a of AXES) v[a] /= n;
  return v;
}

// ─── Config ──────────────────────────────────────────────────────────
export interface SimConfig {
  N: number;
  coercion: number;
  beliefInfluence: number;
  mutationRate: number;
  ritualPeriod: number;
  ritualBonus: number;
  prestigeAlpha: number;
  clusterThreshold: number;
  clusterUpdateFreq: number;
  baseSuccessThresh: number;
  socialK: number;
  socialP: number;
  seed: number;
}

export const DEFAULT_CONFIG: SimConfig = {
  N: 80,
  coercion: 0.0,
  beliefInfluence: 0.15,
  mutationRate: 0.08,
  ritualPeriod: 50,
  ritualBonus: 0.10,
  prestigeAlpha: 0.20,
  clusterThreshold: 0.4,
  clusterUpdateFreq: 25,
  baseSuccessThresh: 0.58,
  socialK: 4,
  socialP: 0.1,
  seed: 42,
};

// ─── Presets ─────────────────────────────────────────────────────────
export interface Preset {
  name: string;
  description: string;
  config: Partial<SimConfig>;
}

export const PRESETS: Preset[] = [
  {
    name: "Ancient Polytheism",
    description: "Low coercion, high diversity — a pluralistic landscape of competing cults and pantheons",
    config: { coercion: 0.05, beliefInfluence: 0.10, mutationRate: 0.12, clusterThreshold: 0.5, N: 100 },
  },
  {
    name: "Imperial Monotheism",
    description: "High coercion — state-enforced religious homogeneity drives a winner-take-all phase transition",
    config: { coercion: 0.85, beliefInfluence: 0.25, mutationRate: 0.03, prestigeAlpha: 0.35, N: 100 },
  },
  {
    name: "Reformation Schism",
    description: "Moderate coercion with high mutation — doctrinal drift fractures a dominant tradition",
    config: { coercion: 0.35, beliefInfluence: 0.20, mutationRate: 0.20, clusterThreshold: 0.35, N: 80 },
  },
  {
    name: "Ritual-Stabilized Society",
    description: "Strong ritual bonding reduces churn and stabilizes existing traditions against drift",
    config: { coercion: 0.15, ritualPeriod: 15, ritualBonus: 0.25, mutationRate: 0.06, N: 80 },
  },
  {
    name: "Charismatic Prophet",
    description: "High prestige amplification — a single influential leader reshapes the belief landscape",
    config: { coercion: 0.20, prestigeAlpha: 0.60, beliefInfluence: 0.30, mutationRate: 0.10, N: 60 },
  },
  {
    name: "Syncretism (Silk Road)",
    description: "High social rewiring + moderate influence — traditions blend at cultural crossroads",
    config: { coercion: 0.10, socialP: 0.4, beliefInfluence: 0.20, mutationRate: 0.08, clusterThreshold: 0.55, N: 100 },
  },
];

// ─── Agent ───────────────────────────────────────────────────────────
export interface Agent {
  id: number;
  belief: Vec;
  prestige: number;
  clusterId: number;
  // 2D projection for visualization (PCA-like)
  x: number;
  y: number;
}

// ─── Cluster / Centroid ──────────────────────────────────────────────
export interface Cluster {
  centroid: Vec;
  agentIds: number[];
  label: string;
  // 2D projection
  cx: number;
  cy: number;
}

// ─── Snapshot (what the UI reads each tick) ───────────────────────────
export interface Snapshot {
  t: number;
  agents: Agent[];
  clusters: Cluster[];
  nEff: number;
  entropy: number;
  dominance: number;
  maxPrestige: number;
  phase: "polytheistic" | "transitional" | "monotheistic";
}

// ─── Context ─────────────────────────────────────────────────────────
const TASKS = ["forage", "warn", "trade", "mourn", "build", "raid"] as const;
const ROLES = ["leader", "hunter", "healer", "crafter"] as const;
const PLACES = ["camp", "river", "forest", "market", "grave"] as const;
const TODS = ["dawn", "noon", "dusk", "night"] as const;

function sampleContext(rng: RNG): Vec {
  const task = rng.choice([...TASKS]);
  const role = rng.choice([...ROLES]);
  const place = rng.choice([...PLACES]);
  const tod = rng.choice([...TODS]);

  const v = zeroVec();
  if (task === "forage") { v.care += 0.4; v.power += 0.3; v.nature += 0.1; }
  else if (task === "warn") { v.authority += 0.4; v.wisdom += 0.2; v.justice += 0.2; }
  else if (task === "trade") { v.justice += 0.3; v.authority += 0.2; v.care += 0.2; }
  else if (task === "mourn") { v.death += 0.5; v.transcendence += 0.2; v.wisdom += 0.2; }
  else if (task === "build") { v.creation += 0.4; v.order += 0.2; v.power += 0.2; }
  else if (task === "raid") { v.war += 0.3; v.justice += 0.3; v.power += 0.2; }

  if (role === "leader") v.authority += 0.3;
  else if (role === "healer") { v.care += 0.3; v.wisdom += 0.2; }
  else if (role === "hunter") { v.power += 0.3; v.nature += 0.2; }
  else if (role === "crafter") v.creation += 0.3;

  if (place === "grave") { v.death += 0.3; v.transcendence += 0.2; }
  else if (place === "market") v.justice += 0.3;
  else if (place === "river") v.nature += 0.2;

  if (tod === "dusk" || tod === "night") v.transcendence += 0.2;
  else v.order += 0.1;

  const n = vecNorm(v) || 1;
  for (const a of AXES) v[a] /= n;
  return v;
}

// ─── 2D Projection ──────────────────────────────────────────────────
// Two orthogonal projection axes chosen to maximally separate
// the most interesting theological contrasts for visualization.
const PROJ_A: Vec = (() => {
  const v = zeroVec();
  // axis 1: Dominion (authority/power/order/war) vs Nurture (care/nature/fertility/creation)
  v.authority = 0.45; v.power = 0.40; v.order = 0.30; v.war = 0.35;
  v.care = -0.40; v.nature = -0.35; v.fertility = -0.30; v.creation = -0.25;
  const n = vecNorm(v) || 1;
  for (const a of AXES) v[a] /= n;
  return v;
})();

const PROJ_B: Vec = (() => {
  const v = zeroVec();
  // axis 2: Transcendence (transcendence/wisdom/justice) vs Chthonic (death/fertility/nature)
  v.transcendence = 0.50; v.wisdom = 0.40; v.justice = 0.35;
  v.death = -0.45; v.fertility = -0.25; v.war = -0.20;
  const n = vecNorm(v) || 1;
  for (const a of AXES) v[a] /= n;
  return v;
})();

function project2D(v: Vec): [number, number] {
  return [dot(v, PROJ_A), dot(v, PROJ_B)];
}

// ─── Deity name generator for centroids ──────────────────────────────
function labelCentroid(centroid: Vec): string {
  // Find the closest deity prior
  let bestName = "Unknown";
  let bestSim = -Infinity;
  for (const [name, prior] of Object.entries(DEITY_PRIORS)) {
    const sim = cosine(centroid, prior);
    if (sim > bestSim) { bestSim = sim; bestName = name; }
  }
  return bestName.charAt(0).toUpperCase() + bestName.slice(1);
}

// ─── SwarmKernel ─────────────────────────────────────────────────────
export class SwarmKernel {
  cfg: SimConfig;
  rng: RNG;
  agents: Agent[];
  clusters: Cluster[];
  socialGraph: Map<number, number[]>;
  t: number;

  constructor(cfg: SimConfig) {
    this.cfg = { ...cfg };
    this.rng = new RNG(cfg.seed);
    this.t = 0;
    this.agents = [];
    this.clusters = [];
    this.socialGraph = new Map();
    this._initAgents();
    this._buildSocialNetwork();
    this._updateClusters();
  }

  private _initAgents() {
    const deityNames = Object.keys(DEITY_PRIORS);
    for (let i = 0; i < this.cfg.N; i++) {
      // Initialize belief from 1-2 deity priors + noise
      const k = this.rng.randInt(1, 3);
      const chosen = this.rng.sample(deityNames, k);
      const belief = zeroVec();
      for (const d of chosen) addScaled(belief, DEITY_PRIORS[d], 1 / k);
      const jittered = jitterVec(belief, this.rng, 0.15);
      const [x, y] = project2D(jittered);
      this.agents.push({ id: i, belief: jittered, prestige: 1.0, clusterId: 0, x, y });
    }
  }

  private _buildSocialNetwork() {
    const N = this.cfg.N;
    const k = Math.floor(this.cfg.socialK / 2);
    this.socialGraph = new Map();
    for (let i = 0; i < N; i++) this.socialGraph.set(i, []);

    // Ring lattice
    for (let i = 0; i < N; i++) {
      for (let j = 1; j <= k; j++) {
        const nb = (i + j) % N;
        this.socialGraph.get(i)!.push(nb);
        this.socialGraph.get(nb)!.push(i);
      }
    }
    // Rewire (Watts-Strogatz)
    for (let i = 0; i < N; i++) {
      const neighbors = this.socialGraph.get(i)!;
      for (let j = 0; j < neighbors.length; j++) {
        if (this.rng.random() < this.cfg.socialP) {
          const old = neighbors[j];
          // Remove old edge
          neighbors[j] = -1; // mark for replacement
          const oldNb = this.socialGraph.get(old)!;
          const idx = oldNb.indexOf(i);
          if (idx >= 0) oldNb.splice(idx, 1);
          // Pick new random neighbor
          let newNb: number;
          do { newNb = this.rng.randInt(0, N); } while (newNb === i || neighbors.includes(newNb));
          neighbors[j] = newNb;
          this.socialGraph.get(newNb)!.push(i);
        }
      }
    }
  }

  private _updateClusters() {
    const centroids: Vec[] = [];
    const clusterIds: number[][] = [];

    for (const agent of this.agents) {
      if (centroids.length === 0) {
        centroids.push({ ...agent.belief });
        clusterIds.push([agent.id]);
        agent.clusterId = 0;
        continue;
      }

      let bestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < centroids.length; i++) {
        const d = 1 - cosine(agent.belief, centroids[i]);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }

      if (bestDist < this.cfg.clusterThreshold) {
        clusterIds[bestIdx].push(agent.id);
        agent.clusterId = bestIdx;
      } else {
        agent.clusterId = centroids.length;
        centroids.push({ ...agent.belief });
        clusterIds.push([agent.id]);
      }
    }

    // Recalculate centroids using prestige-weighted average (Definition 5)
    this.clusters = [];
    for (let i = 0; i < centroids.length; i++) {
      if (clusterIds[i].length === 0) continue;
      const c = zeroVec();
      let totalWeight = 0;
      for (const aid of clusterIds[i]) {
        const w = this.agents[aid].prestige;
        addScaled(c, this.agents[aid].belief, w);
        totalWeight += w;
      }
      const sc = totalWeight > 0 ? scaleVec(c, 1 / totalWeight) : c;
      const [cx, cy] = project2D(sc);
      this.clusters.push({
        centroid: sc,
        agentIds: clusterIds[i],
        label: labelCentroid(sc),
        cx, cy,
      });
    }
    // Re-assign cluster IDs to agents
    for (let ci = 0; ci < this.clusters.length; ci++) {
      for (const aid of this.clusters[ci].agentIds) {
        this.agents[aid].clusterId = ci;
      }
    }
  }

  /** Run one simulation step */
  step() {
    this.t++;
    const ctx = sampleContext(this.rng);
    const isRitual = this.t % this.cfg.ritualPeriod === 0;

    // Pick speaker weighted by prestige
    const weights = this.agents.map(a => a.prestige);
    const speaker = this.rng.weightedChoice(this.agents, weights);

    // Pick hearers from social network
    const neighbors = this.socialGraph.get(speaker.id) || [];
    let hearers: Agent[];
    if (neighbors.length === 0) {
      hearers = this.rng.sample(this.agents.filter(a => a.id !== speaker.id), 2);
    } else if (this.cfg.coercion > 0) {
      const nWeights = neighbors.map(nid => {
        const sim = cosine(speaker.belief, this.agents[nid].belief);
        return Math.exp(sim * (1 + 9 * this.cfg.coercion));
      });
      const h1 = this.rng.weightedChoice(neighbors, nWeights);
      hearers = [this.agents[h1]];
    } else {
      const chosen = this.rng.sample(neighbors, Math.min(2, neighbors.length));
      hearers = chosen.map(id => this.agents[id]);
    }

    // Interaction: check if communication succeeds
    const sims = hearers.map(h => cosine(ctx, h.belief));
    const avgSim = sims.reduce((a, b) => a + b, 0) / sims.length;
    const thresh = this.cfg.baseSuccessThresh + (isRitual ? this.cfg.ritualBonus : 0);
    const success = avgSim >= thresh;

    // Learning: shift beliefs toward context on success, away on failure
    const lr = success ? 0.08 : -0.02;
    const allParticipants = [speaker, ...hearers];
    for (const agent of allParticipants) {
      // Blend context with coercion pressure toward dominant centroid
      const target = zeroVec();
      addScaled(target, ctx, 1);

      if (this.cfg.coercion > 0 && this.clusters.length > 0) {
        // Find largest cluster centroid
        let largest = this.clusters[0];
        for (const cl of this.clusters) {
          if (cl.agentIds.length > largest.agentIds.length) largest = cl;
        }
        addScaled(target, largest.centroid, this.cfg.coercion * 0.3);
      }

      addScaled(agent.belief, target, lr * (1 + this.cfg.beliefInfluence));

      // Renormalize
      const n = vecNorm(agent.belief) || 1;
      for (const a of AXES) agent.belief[a] /= n;

      // Update 2D projection
      const [x, y] = project2D(agent.belief);
      agent.x = x;
      agent.y = y;
    }

    // Prestige update
    for (const agent of allParticipants) {
      const delta = success ? this.cfg.prestigeAlpha : -this.cfg.prestigeAlpha * 0.3;
      agent.prestige = Math.max(0.1, Math.min(10, agent.prestige * (1 + delta)));
    }

    // Mutation: random belief drift
    for (const agent of allParticipants) {
      if (this.rng.random() < this.cfg.mutationRate) {
        const noise = jitterVec(agent.belief, this.rng, 0.05);
        agent.belief = noise;
        const [x, y] = project2D(agent.belief);
        agent.x = x;
        agent.y = y;
      }
    }

    // Clustering update
    if (this.t % this.cfg.clusterUpdateFreq === 0) {
      this._updateClusters();
    }
  }

  /** Run multiple steps */
  run(steps: number) {
    for (let i = 0; i < steps; i++) this.step();
  }

  /** Get current snapshot for visualization */
  snapshot(): Snapshot {
    const nEff = this.clusters.length;
    const sizes = this.clusters.map(c => c.agentIds.length);
    const total = this.cfg.N;
    const dominance = sizes.length > 0 ? Math.max(...sizes) / total : 0;

    // Shannon entropy of cluster distribution
    let entropy = 0;
    for (const s of sizes) {
      if (s > 0) {
        const p = s / total;
        entropy -= p * Math.log2(p);
      }
    }

    const maxPrestige = Math.max(...this.agents.map(a => a.prestige));

    let phase: Snapshot["phase"];
    if (nEff <= 2 && dominance > 0.5) phase = "monotheistic";
    else if (nEff >= 5) phase = "polytheistic";
    else phase = "transitional";

    return {
      t: this.t,
      agents: this.agents,
      clusters: this.clusters,
      nEff,
      entropy,
      dominance,
      maxPrestige,
      phase,
    };
  }

  /** Live-update config without resetting */
  updateConfig(partial: Partial<SimConfig>) {
    Object.assign(this.cfg, partial);
  }

  /** Full reset with new config */
  reset(cfg?: Partial<SimConfig>) {
    if (cfg) Object.assign(this.cfg, cfg);
    this.rng = new RNG(this.cfg.seed);
    this.t = 0;
    this.agents = [];
    this.clusters = [];
    this.socialGraph = new Map();
    this._initAgents();
    this._buildSocialNetwork();
    this._updateClusters();
  }

  /**
   * Inject a "Prophet" — a charismatic agent with a novel belief vector
   * that pulls nearby agents toward it. Models Section 4.3 of the paper:
   * punctuated shifts via revelation/prophecy.
   */
  injectProphet() {
    // Create a novel belief in an underrepresented region of belief space
    const prophet = this.agents[this.rng.randInt(0, this.agents.length)];
    // Generate a radical new belief (far from current centroids)
    const novelBelief = rndUnitVec(this.rng);
    prophet.belief = novelBelief;
    prophet.prestige = 8.0; // Very high prestige — charismatic leader
    const [x, y] = project2D(novelBelief);
    prophet.x = x;
    prophet.y = y;

    // Also pull 15% of nearby agents toward the prophet's belief
    const pullCount = Math.floor(this.cfg.N * 0.15);
    const sorted = [...this.agents]
      .filter(a => a.id !== prophet.id)
      .sort((a, b) => cosine(b.belief, prophet.belief) - cosine(a.belief, prophet.belief));

    for (let i = 0; i < Math.min(pullCount, sorted.length); i++) {
      const agent = sorted[i];
      // Shift 40% toward prophet's belief
      for (const ax of AXES) {
        agent.belief[ax] = agent.belief[ax] * 0.6 + prophet.belief[ax] * 0.4;
      }
      const n = vecNorm(agent.belief) || 1;
      for (const ax of AXES) agent.belief[ax] /= n;
      const [ax2, ay2] = project2D(agent.belief);
      agent.x = ax2;
      agent.y = ay2;
    }

    this._updateClusters();
  }

  /**
   * Environmental shock — a sudden external event that perturbs all agents'
   * beliefs along specific theological axes. Models famine, plague, conquest.
   */
  environmentalShock(type: "war" | "plague" | "abundance" | "contact") {
    const shockVec = zeroVec();
    switch (type) {
      case "war":
        shockVec.war = 0.4; shockVec.death = 0.3; shockVec.power = 0.2;
        shockVec.care = -0.2; shockVec.nature = -0.1;
        break;
      case "plague":
        shockVec.death = 0.5; shockVec.transcendence = 0.3;
        shockVec.power = -0.2; shockVec.order = -0.2;
        break;
      case "abundance":
        shockVec.fertility = 0.4; shockVec.nature = 0.3; shockVec.care = 0.2;
        shockVec.war = -0.2; shockVec.death = -0.2;
        break;
      case "contact":
        // Cultural contact — random perturbation simulating foreign influence
        for (const ax of AXES) shockVec[ax] = this.rng.gauss(0, 0.15);
        break;
    }

    // Apply shock to all agents with varying intensity
    for (const agent of this.agents) {
      const intensity = 0.15 + this.rng.random() * 0.15; // 15-30% shift
      addScaled(agent.belief, shockVec, intensity);
      const n = vecNorm(agent.belief) || 1;
      for (const ax of AXES) agent.belief[ax] /= n;
      const [x, y] = project2D(agent.belief);
      agent.x = x;
      agent.y = y;
    }

    this._updateClusters();
  }

  /**
   * Force syncretism — merge the two closest clusters.
   * Models Section 4.1: fusion when traditions come into sustained contact.
   */
  forceSyncretism() {
    if (this.clusters.length < 2) return;

    // Find two closest centroids
    let bestI = 0, bestJ = 1, bestSim = -Infinity;
    for (let i = 0; i < this.clusters.length; i++) {
      for (let j = i + 1; j < this.clusters.length; j++) {
        const sim = cosine(this.clusters[i].centroid, this.clusters[j].centroid);
        if (sim > bestSim) { bestSim = sim; bestI = i; bestJ = j; }
      }
    }

    // Merge: shift all agents in cluster j toward cluster i's centroid
    const targetCentroid = this.clusters[bestI].centroid;
    for (const aid of this.clusters[bestJ].agentIds) {
      const agent = this.agents[aid];
      for (const ax of AXES) {
        agent.belief[ax] = agent.belief[ax] * 0.4 + targetCentroid[ax] * 0.6;
      }
      const n = vecNorm(agent.belief) || 1;
      for (const ax of AXES) agent.belief[ax] /= n;
      const [x, y] = project2D(agent.belief);
      agent.x = x;
      agent.y = y;
    }

    this._updateClusters();
  }
}
