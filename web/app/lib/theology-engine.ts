/**
 * Theology Engine — Generative Agent-Based Model (GABM) Layer
 *
 * Treats emergent deity centroids as first-class mathematical objects.
 * Supports:
 *   - Extraction of deity objects from simulation state
 *   - Meta-centroid fusion with explicit mixing weights
 *   - Interpolation between deities (slerp on the unit sphere)
 *   - Mutation (controlled perturbation of deity vectors)
 *   - Deterministic theological description generation from geometry
 *   - LLM adapter interface for richer generation when available
 *
 * Key principle: gods are centroids, fusion is geometric, generation
 * is structural, belief remains human.
 */

import { AXES, type Axis, type Vec, type Cluster, type Snapshot } from "./simulation";

// ─── Deity Object ────────────────────────────────────────────────────

export interface DeityObject {
  /** Unique identifier */
  id: string;
  /** Human-readable name (from nearest prior, or generated) */
  name: string;
  /** 12D belief vector (unit-normalized) */
  vector: Vec;
  /** Number of adherents in the originating cluster */
  adherents: number;
  /** Prestige-weighted vector (if available) */
  weightedVector?: Vec;
  /** Source: 'emergent' from simulation, 'prior' from DEITY_PRIORS, 'fused', 'interpolated', 'mutated' */
  source: "emergent" | "prior" | "fused" | "interpolated" | "mutated";
  /** Parent deity IDs (for fused/interpolated/mutated deities) */
  parentIds?: string[];
  /** Mixing weights used in fusion (parallel to parentIds) */
  mixingWeights?: number[];
  /** Top 3 dominant axes */
  dominantAxes: [Axis, Axis, Axis];
  /** Top 3 recessive axes */
  recessiveAxes: [Axis, Axis, Axis];
  /** Theological profile (generated description) */
  profile: TheologicalProfile;
}

export interface TheologicalProfile {
  /** One-line archetype description */
  archetype: string;
  /** Domain of influence */
  domains: string[];
  /** Worship style */
  worshipStyle: string;
  /** Moral emphasis */
  moralEmphasis: string;
  /** Cosmological role */
  cosmologicalRole: string;
  /** Relationship to death/afterlife */
  eschatology: string;
  /** Relationship to nature */
  natureRelation: string;
  /** Authority structure implied */
  authorityStructure: string;
}

// ─── Vector Utilities ────────────────────────────────────────────────

function zeroVec(): Vec {
  const v = {} as Vec;
  for (const a of AXES) v[a] = 0;
  return v;
}

function copyVec(v: Vec): Vec {
  const c = {} as Vec;
  for (const a of AXES) c[a] = v[a];
  return c;
}

function vecNorm(v: Vec): number {
  let s = 0;
  for (const a of AXES) s += v[a] * v[a];
  return Math.sqrt(s);
}

function normalize(v: Vec): Vec {
  const n = vecNorm(v) || 1;
  const r = {} as Vec;
  for (const a of AXES) r[a] = v[a] / n;
  return r;
}

function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (const k of AXES) s += a[k] * b[k];
  return s;
}

function cosine(a: Vec, b: Vec): number {
  const na = vecNorm(a);
  const nb = vecNorm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

function sortedAxes(v: Vec): Axis[] {
  return [...AXES].sort((a, b) => v[b] - v[a]);
}

function topN(v: Vec, n: number): Axis[] {
  return sortedAxes(v).slice(0, n);
}

function bottomN(v: Vec, n: number): Axis[] {
  return sortedAxes(v).slice(-n).reverse();
}

// ─── Extraction ──────────────────────────────────────────────────────

/**
 * Extract deity objects from a simulation snapshot.
 * Each cluster with >= minAdherents agents becomes a DeityObject.
 */
export function extractDeities(
  snapshot: Snapshot,
  minAdherents: number = 2,
): DeityObject[] {
  return snapshot.clusters
    .filter((c) => c.agentIds.length >= minAdherents)
    .map((cluster, i) => {
      const vec = normalize(cluster.centroid);
      const dominant = topN(vec, 3) as [Axis, Axis, Axis];
      const recessive = bottomN(vec, 3) as [Axis, Axis, Axis];

      return {
        id: `emergent-${snapshot.t}-${i}`,
        name: cluster.label,
        vector: vec,
        adherents: cluster.agentIds.length,
        source: "emergent" as const,
        dominantAxes: dominant,
        recessiveAxes: recessive,
        profile: generateProfile(vec, dominant, recessive),
      };
    });
}

// ─── Fusion ──────────────────────────────────────────────────────────

/**
 * Fuse multiple deity centroids into a meta-centroid.
 *
 * D* = Σ αⱼ Dⱼ  with Σ αⱼ = 1
 *
 * The result is a second-order semantic attractor — not "the true god"
 * but a geometric synthesis of theological structures.
 */
export function fuseDeities(
  deities: DeityObject[],
  weights?: number[],
  name?: string,
): DeityObject {
  const n = deities.length;
  // Default: equal weights
  const w = weights ?? deities.map(() => 1 / n);

  // Normalize weights to sum to 1
  const wSum = w.reduce((a, b) => a + b, 0);
  const alpha = w.map((x) => x / wSum);

  // Compute weighted sum
  const fused = zeroVec();
  for (let j = 0; j < n; j++) {
    for (const a of AXES) {
      fused[a] += alpha[j] * deities[j].vector[a];
    }
  }

  const vec = normalize(fused);
  const dominant = topN(vec, 3) as [Axis, Axis, Axis];
  const recessive = bottomN(vec, 3) as [Axis, Axis, Axis];

  // Generate name from dominant axes if not provided
  const fusedName =
    name ??
    `${capitalize(dominant[0])}-${capitalize(dominant[1])} Synthesis`;

  return {
    id: `fused-${Date.now()}`,
    name: fusedName,
    vector: vec,
    adherents: deities.reduce((s, d) => s + d.adherents, 0),
    source: "fused",
    parentIds: deities.map((d) => d.id),
    mixingWeights: alpha,
    dominantAxes: dominant,
    recessiveAxes: recessive,
    profile: generateProfile(vec, dominant, recessive),
  };
}

// ─── Interpolation (Slerp) ───────────────────────────────────────────

/**
 * Spherical linear interpolation between two deity vectors.
 * t=0 → deity A, t=1 → deity B, t=0.5 → midpoint on the great circle.
 */
export function interpolateDeities(
  a: DeityObject,
  b: DeityObject,
  t: number,
  name?: string,
): DeityObject {
  const cosTheta = dot(a.vector, b.vector);
  const theta = Math.acos(Math.max(-1, Math.min(1, cosTheta)));

  let vec: Vec;
  if (Math.abs(theta) < 1e-6) {
    // Vectors are nearly identical — linear interpolation
    vec = zeroVec();
    for (const ax of AXES) {
      vec[ax] = (1 - t) * a.vector[ax] + t * b.vector[ax];
    }
  } else {
    // Slerp
    const sinTheta = Math.sin(theta);
    const wa = Math.sin((1 - t) * theta) / sinTheta;
    const wb = Math.sin(t * theta) / sinTheta;
    vec = zeroVec();
    for (const ax of AXES) {
      vec[ax] = wa * a.vector[ax] + wb * b.vector[ax];
    }
  }

  vec = normalize(vec);
  const dominant = topN(vec, 3) as [Axis, Axis, Axis];
  const recessive = bottomN(vec, 3) as [Axis, Axis, Axis];

  const interpName =
    name ?? `${a.name}↔${b.name} (t=${t.toFixed(2)})`;

  return {
    id: `interp-${Date.now()}`,
    name: interpName,
    vector: vec,
    adherents: Math.round((1 - t) * a.adherents + t * b.adherents),
    source: "interpolated",
    parentIds: [a.id, b.id],
    mixingWeights: [1 - t, t],
    dominantAxes: dominant,
    recessiveAxes: recessive,
    profile: generateProfile(vec, dominant, recessive),
  };
}

// ─── Mutation ────────────────────────────────────────────────────────

/**
 * Mutate a deity vector by perturbing it in a specific direction.
 * magnitude controls how far; direction can be a specific axis to amplify.
 */
export function mutateDeity(
  deity: DeityObject,
  magnitude: number = 0.1,
  direction?: Partial<Vec>,
  name?: string,
): DeityObject {
  const mutated = copyVec(deity.vector);

  if (direction) {
    // Directed mutation
    for (const a of AXES) {
      mutated[a] += magnitude * (direction[a] ?? 0);
    }
  } else {
    // Random mutation (seeded by current time for determinism in a session)
    const seed = Date.now();
    for (let i = 0; i < AXES.length; i++) {
      // Simple deterministic pseudo-random based on seed + axis index
      const hash = Math.sin(seed * 9301 + i * 49297) * 49297;
      const noise = (hash - Math.floor(hash)) * 2 - 1; // [-1, 1]
      mutated[AXES[i]] += magnitude * noise;
    }
  }

  const vec = normalize(mutated);
  const dominant = topN(vec, 3) as [Axis, Axis, Axis];
  const recessive = bottomN(vec, 3) as [Axis, Axis, Axis];

  const mutName = name ?? `${deity.name} (mutated)`;

  return {
    id: `mutated-${Date.now()}`,
    name: mutName,
    vector: vec,
    adherents: deity.adherents,
    source: "mutated",
    parentIds: [deity.id],
    dominantAxes: dominant,
    recessiveAxes: recessive,
    profile: generateProfile(vec, dominant, recessive),
  };
}

// ─── Interpolation Sweep ─────────────────────────────────────────────

/**
 * Generate a sweep of interpolated deities between two endpoints.
 * Useful for exploring the "theological gradient" between traditions.
 */
export function interpolationSweep(
  a: DeityObject,
  b: DeityObject,
  steps: number = 10,
): DeityObject[] {
  const result: DeityObject[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    result.push(interpolateDeities(a, b, t));
  }
  return result;
}

// ─── Similarity Matrix ───────────────────────────────────────────────

/**
 * Compute pairwise cosine similarity matrix for a set of deities.
 */
export function similarityMatrix(
  deities: DeityObject[],
): { labels: string[]; matrix: number[][] } {
  const n = deities.length;
  const labels = deities.map((d) => d.name);
  const matrix: number[][] = [];

  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      row.push(cosine(deities[i].vector, deities[j].vector));
    }
    matrix.push(row);
  }

  return { labels, matrix };
}

// ─── Theological Profile Generation ──────────────────────────────────

const AXIS_DOMAINS: Record<Axis, string[]> = {
  authority: ["kingship", "law", "sovereignty", "governance", "hierarchy"],
  transcendence: ["the heavens", "the divine realm", "cosmic order", "the infinite", "the sacred"],
  care: ["healing", "compassion", "nurture", "protection of the weak", "mercy"],
  justice: ["judgment", "moral law", "retribution", "fairness", "cosmic balance"],
  wisdom: ["knowledge", "prophecy", "insight", "learning", "revelation"],
  power: ["strength", "might", "dominion", "force", "sovereignty"],
  fertility: ["abundance", "harvest", "procreation", "growth", "renewal"],
  war: ["battle", "conquest", "valor", "defense", "martial honor"],
  death: ["the underworld", "ancestral spirits", "mortality", "passage", "transformation"],
  creation: ["the cosmos", "craftsmanship", "genesis", "world-building", "primordial order"],
  nature: ["the wild", "seasons", "animals", "the earth", "natural cycles"],
  order: ["civilization", "ritual", "structure", "cosmic law", "harmony"],
};

const AXIS_WORSHIP: Record<Axis, string> = {
  authority: "formal liturgy with hierarchical priesthood",
  transcendence: "contemplative meditation and mystical practice",
  care: "communal healing rituals and acts of charity",
  justice: "oath-taking ceremonies and judicial rites",
  wisdom: "scholarly study and oracular consultation",
  power: "sacrificial offerings and displays of devotion",
  fertility: "seasonal festivals and agricultural rites",
  war: "martial ceremonies and victory celebrations",
  death: "funerary rites and ancestor veneration",
  creation: "creation narratives and artisan dedications",
  nature: "outdoor worship and nature communion",
  order: "calendrical observance and structured ceremony",
};

const AXIS_MORAL: Record<Axis, string> = {
  authority: "obedience and respect for hierarchy",
  transcendence: "spiritual purity and detachment from the material",
  care: "compassion and protection of the vulnerable",
  justice: "fairness, truth-telling, and accountability",
  wisdom: "pursuit of knowledge and discernment",
  power: "strength, courage, and self-mastery",
  fertility: "generosity, hospitality, and abundance-sharing",
  war: "honor, loyalty, and sacrifice for the group",
  death: "acceptance of mortality and reverence for ancestors",
  creation: "creativity, stewardship, and world-building",
  nature: "harmony with the natural world and ecological balance",
  order: "discipline, ritual observance, and social cohesion",
};

const AXIS_COSMOLOGY: Record<Axis, string> = {
  authority: "a divine sovereign ruling from above",
  transcendence: "an ineffable presence beyond material reality",
  care: "a nurturing parent sustaining all life",
  justice: "a cosmic judge weighing all deeds",
  wisdom: "an all-knowing mind pervading reality",
  power: "a supreme force shaping the world through will",
  fertility: "a generative principle from which all life springs",
  war: "a champion defending cosmic order against chaos",
  death: "a guardian of the threshold between worlds",
  creation: "a primordial craftsman who shaped the cosmos",
  nature: "an immanent spirit dwelling in all living things",
  order: "an architect of the laws governing existence",
};

const AXIS_ESCHATOLOGY: Record<Axis, string> = {
  authority: "judgment before a divine throne",
  transcendence: "dissolution into the infinite",
  care: "reunion with loved ones in a place of comfort",
  justice: "reward or punishment proportional to deeds",
  wisdom: "attainment of ultimate understanding",
  power: "eternal glory for the worthy",
  fertility: "cyclical rebirth and renewal",
  war: "a warrior's paradise for the valiant",
  death: "peaceful rest in the ancestral realm",
  creation: "participation in an ongoing act of creation",
  nature: "return to the earth and the cycle of life",
  order: "integration into the eternal cosmic pattern",
};

const AXIS_NATURE_REL: Record<Axis, string> = {
  authority: "nature as subject to divine command",
  transcendence: "nature as a veil over deeper reality",
  care: "nature as a garden to be tended",
  justice: "nature as governed by moral law",
  wisdom: "nature as a book of divine knowledge",
  power: "nature as raw material for divine will",
  fertility: "nature as the body of the divine",
  war: "nature as a battleground of cosmic forces",
  death: "nature as the cycle of death and rebirth",
  creation: "nature as the masterwork of the creator",
  nature: "nature as sacred and self-sufficient",
  order: "nature as an expression of cosmic harmony",
};

const AXIS_AUTHORITY_STRUCT: Record<Axis, string> = {
  authority: "centralized hierarchy with supreme pontiff",
  transcendence: "decentralized mystic circles and hermits",
  care: "egalitarian community with rotating leadership",
  justice: "council of elders and judges",
  wisdom: "meritocratic academy of scholars",
  power: "strongman leadership with warrior-priests",
  fertility: "matriarchal or seasonal priesthood",
  war: "military-religious order",
  death: "necromantic priesthood and death-cult hierarchy",
  creation: "guild of sacred artisans",
  nature: "shamanic tradition with no fixed hierarchy",
  order: "bureaucratic temple administration",
};

function generateArchetype(dominant: [Axis, Axis, Axis], vec: Vec): string {
  const [a, b, c] = dominant;
  const archetypes: Record<string, string> = {
    "authority-power": "Sovereign Storm-King",
    "authority-justice": "Divine Lawgiver",
    "authority-transcendence": "Celestial Emperor",
    "authority-wisdom": "All-Seeing Ruler",
    "authority-order": "Cosmic Architect-King",
    "transcendence-wisdom": "Ineffable Oracle",
    "transcendence-care": "Compassionate Infinite",
    "transcendence-creation": "Primordial Dreamer",
    "transcendence-nature": "Spirit of the Wild Cosmos",
    "care-wisdom": "Gentle Sage",
    "care-fertility": "Great Mother",
    "care-nature": "Earth Healer",
    "justice-power": "Righteous Champion",
    "justice-wisdom": "Cosmic Judge",
    "justice-war": "Holy Avenger",
    "wisdom-creation": "Divine Artificer",
    "wisdom-death": "Keeper of Hidden Knowledge",
    "wisdom-nature": "Forest Philosopher",
    "power-war": "War God",
    "power-creation": "World-Forger",
    "power-death": "Lord of the Underworld",
    "fertility-nature": "Green Deity",
    "fertility-creation": "Life-Giver",
    "war-death": "Psychopomp Warrior",
    "war-power": "Thunder Champion",
    "death-transcendence": "Veiled Threshold Guardian",
    "death-wisdom": "Ancestor-Sage",
    "creation-order": "Cosmic Craftsman",
    "nature-order": "Harmony Spirit",
    "nature-death": "Cycle Keeper",
    "order-authority": "Supreme Legislator",
  };

  const key1 = `${a}-${b}`;
  const key2 = `${b}-${a}`;
  return archetypes[key1] ?? archetypes[key2] ?? `${capitalize(a)}-${capitalize(b)} Deity`;
}

function generateProfile(
  vec: Vec,
  dominant: [Axis, Axis, Axis],
  recessive: [Axis, Axis, Axis],
): TheologicalProfile {
  const [d1, d2, d3] = dominant;

  return {
    archetype: generateArchetype(dominant, vec),
    domains: [
      AXIS_DOMAINS[d1][Math.floor(vec[d1] * 4.99)],
      AXIS_DOMAINS[d2][Math.floor(vec[d2] * 4.99)],
      AXIS_DOMAINS[d3][Math.floor(vec[d3] * 4.99)],
    ],
    worshipStyle: AXIS_WORSHIP[d1],
    moralEmphasis: AXIS_MORAL[d2],
    cosmologicalRole: AXIS_COSMOLOGY[d1],
    eschatology: AXIS_ESCHATOLOGY[d2],
    natureRelation: AXIS_NATURE_REL[d3],
    authorityStructure: AXIS_AUTHORITY_STRUCT[d1],
  };
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ─── LLM Adapter Interface ──────────────────────────────────────────

/**
 * Interface for plugging in an LLM to generate richer theological
 * descriptions from deity vectors. The engine works without an LLM
 * (using the deterministic profile generator above), but an LLM
 * adapter can produce scripture, prayers, myths, and doctrinal texts.
 */
export interface TheologyLLMAdapter {
  /**
   * Generate a theological description from a deity vector.
   * The prompt should include the vector components and profile.
   */
  generateDescription(deity: DeityObject): Promise<string>;

  /**
   * Generate a creation myth for a deity.
   */
  generateMyth(deity: DeityObject): Promise<string>;

  /**
   * Generate a prayer or invocation.
   */
  generatePrayer(deity: DeityObject): Promise<string>;

  /**
   * Generate doctrinal tenets.
   */
  generateDoctrine(deity: DeityObject): Promise<string[]>;

  /**
   * Compare two deities and generate a theological analysis.
   */
  compareDeities(a: DeityObject, b: DeityObject): Promise<string>;
}

/**
 * Build a structured prompt from a deity object for use with any LLM.
 */
export function buildDeityPrompt(deity: DeityObject): string {
  const axes = AXES.map((a) => `  ${a}: ${deity.vector[a].toFixed(3)}`).join("\n");
  const profile = deity.profile;

  return `You are a theology engine. Given the following deity vector in a 12-dimensional theological space, generate a rich theological description. This is a mathematical object, not a real deity. Treat it as a plausible god-concept that could have emerged in human history.

DEITY: ${deity.name}
SOURCE: ${deity.source}
${deity.parentIds ? `PARENTS: ${deity.parentIds.join(", ")}` : ""}
${deity.mixingWeights ? `MIXING WEIGHTS: ${deity.mixingWeights.map((w) => w.toFixed(3)).join(", ")}` : ""}

VECTOR COMPONENTS:
${axes}

DOMINANT AXES: ${deity.dominantAxes.join(", ")}
RECESSIVE AXES: ${deity.recessiveAxes.join(", ")}

GENERATED PROFILE:
  Archetype: ${profile.archetype}
  Domains: ${profile.domains.join(", ")}
  Worship: ${profile.worshipStyle}
  Moral emphasis: ${profile.moralEmphasis}
  Cosmological role: ${profile.cosmologicalRole}
  Eschatology: ${profile.eschatology}
  Nature relation: ${profile.natureRelation}
  Authority structure: ${profile.authorityStructure}

Generate a detailed theological description including:
1. A one-paragraph summary of this deity's nature and character
2. Three core doctrinal tenets
3. A brief creation myth fragment
4. A sample prayer or invocation
5. How this deity relates to human suffering

Remember: this is a structural analysis, not a religious claim. The deity is a centroid, not a cause.`;
}

// ─── Export convenience ──────────────────────────────────────────────

export type { Vec, Axis } from "./simulation";
