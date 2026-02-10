/**
 * Braille Lattice — Discrete Semantic Substrate for Belief Space
 *
 * Braille is introduced not as an accessibility layer, but as a discrete
 * semantic substrate that enforces compression, modality invariance, and
 * structural stability in theological belief dynamics.
 *
 * Architecture:
 *   Continuous belief vector (ℝ¹²) → Braille lattice ({0,1}ⁿ)
 *
 * Each theological axis is encoded as a bundle of braille cells:
 *   - 3 cells for polarity (e.g., justice ↔ mercy)
 *   - 2 cells for intensity (low/medium/high/max)
 *   - 1 cell for rigidity (flexible vs dogmatic)
 *   = 6 dots per axis = 1 standard braille cell per axis
 *   = 12 axes × 6 dots = 72-bit braille representation
 *
 * This gives us:
 *   - Centroids as Hamming means on discrete space
 *   - Deities as stable braille configurations
 *   - Drift as cell flips (discrete, countable)
 *   - Natural phase transitions (snap, not slide)
 *   - Punctuated equilibrium for free
 */

import { AXES, type Axis, type Vec } from "./simulation";

// ─── Braille Cell ────────────────────────────────────────────────────

/**
 * A standard braille cell: 6 dots arranged in a 3×2 grid.
 * Dots are numbered 1-6 (standard braille numbering):
 *   1 4
 *   2 5
 *   3 6
 *
 * We encode each theological axis as one braille cell:
 *   Dots 1,2,3 (left column): polarity encoding
 *     - Dot 1: positive polarity active
 *     - Dot 2: negative polarity active
 *     - Dot 3: polarity ambiguity (both active = tension)
 *   Dots 4,5 (right column top): intensity
 *     - 00 = low, 01 = medium, 10 = high, 11 = maximum
 *   Dot 6 (right column bottom): rigidity
 *     - 0 = flexible/fluid, 1 = dogmatic/fixed
 */
export interface BrailleCell {
  dots: [boolean, boolean, boolean, boolean, boolean, boolean];
}

/**
 * A braille lattice point: one cell per theological axis.
 * Total: 12 cells × 6 dots = 72 bits.
 */
export type BrailleLatticePoint = Record<Axis, BrailleCell>;

/**
 * Flat bit-string representation for Hamming distance computation.
 */
export type BitString = boolean[];

// ─── Polarity Pairs ──────────────────────────────────────────────────

/**
 * Each axis has a conceptual opposite. When both are high in a deity
 * vector, the braille encoding captures the tension (dot 3 active).
 */
const POLARITY_PAIRS: Record<Axis, Axis> = {
  authority: "care",
  care: "authority",
  transcendence: "nature",
  nature: "transcendence",
  justice: "fertility",
  fertility: "justice",
  wisdom: "war",
  war: "wisdom",
  power: "death",
  death: "power",
  creation: "order",
  order: "creation",
};

// ─── Encoding ────────────────────────────────────────────────────────

/**
 * Encode a continuous belief vector as a braille lattice point.
 *
 * vec ∈ ℝ¹² → 𝓑 ∈ {0,1}⁷²
 */
export function encodeToBraille(vec: Vec): BrailleLatticePoint {
  const result = {} as BrailleLatticePoint;

  for (const axis of AXES) {
    const value = vec[axis];
    const opposite = POLARITY_PAIRS[axis];
    const oppositeValue = vec[opposite];

    // Polarity encoding (dots 1-3)
    const posActive = value > 0.3;
    const negActive = oppositeValue > value + 0.1;
    const tension = posActive && oppositeValue > 0.3 && Math.abs(value - oppositeValue) < 0.15;

    // Intensity encoding (dots 4-5)
    // Quantize to 4 levels: [0, 0.25) = 00, [0.25, 0.5) = 01, [0.5, 0.75) = 10, [0.75, 1] = 11
    const intensity = Math.min(3, Math.floor(value * 4));
    const dot4 = (intensity & 2) !== 0;
    const dot5 = (intensity & 1) !== 0;

    // Rigidity encoding (dot 6)
    // High value + low variance from neighbors → dogmatic
    // We approximate rigidity as: value > 0.7 (strong conviction)
    const rigid = value > 0.7;

    result[axis] = {
      dots: [posActive, negActive, tension, dot4, dot5, rigid],
    };
  }

  return result;
}

/**
 * Decode a braille lattice point back to an approximate continuous vector.
 * This is lossy — the braille encoding is a compression.
 */
export function decodeFromBraille(lattice: BrailleLatticePoint): Vec {
  const vec = {} as Vec;

  for (const axis of AXES) {
    const cell = lattice[axis];
    const [posActive, negActive, tension, dot4, dot5, rigid] = cell.dots;

    // Reconstruct intensity from dots 4-5
    const intensity = (dot4 ? 2 : 0) + (dot5 ? 1 : 0);
    let value = (intensity + 0.5) / 4; // center of quantization bin

    // Adjust for polarity
    if (!posActive && negActive) {
      value *= 0.3; // suppressed by opposite
    }
    if (tension) {
      value *= 0.85; // tension reduces effective strength
    }

    // Rigidity boost
    if (rigid) {
      value = Math.max(value, 0.75);
    }

    vec[axis] = value;
  }

  // Normalize
  let norm = 0;
  for (const a of AXES) norm += vec[a] * vec[a];
  norm = Math.sqrt(norm) || 1;
  for (const a of AXES) vec[a] /= norm;

  return vec;
}

// ─── Bit-String Operations ───────────────────────────────────────────

/**
 * Flatten a braille lattice point to a 72-bit string.
 */
export function toBitString(lattice: BrailleLatticePoint): BitString {
  const bits: boolean[] = [];
  for (const axis of AXES) {
    for (const dot of lattice[axis].dots) {
      bits.push(dot);
    }
  }
  return bits;
}

/**
 * Reconstruct a braille lattice point from a 72-bit string.
 */
export function fromBitString(bits: BitString): BrailleLatticePoint {
  const result = {} as BrailleLatticePoint;
  let idx = 0;
  for (const axis of AXES) {
    result[axis] = {
      dots: [bits[idx], bits[idx + 1], bits[idx + 2], bits[idx + 3], bits[idx + 4], bits[idx + 5]],
    };
    idx += 6;
  }
  return result;
}

/**
 * Hamming distance between two braille lattice points.
 * Counts the number of bit positions that differ.
 */
export function hammingDistance(a: BrailleLatticePoint, b: BrailleLatticePoint): number {
  const bitsA = toBitString(a);
  const bitsB = toBitString(b);
  let dist = 0;
  for (let i = 0; i < bitsA.length; i++) {
    if (bitsA[i] !== bitsB[i]) dist++;
  }
  return dist;
}

/**
 * Normalized Hamming distance (0 = identical, 1 = maximally different).
 */
export function normalizedHammingDistance(
  a: BrailleLatticePoint,
  b: BrailleLatticePoint,
): number {
  return hammingDistance(a, b) / 72;
}

// ─── Braille Centroid (Hamming Mean) ─────────────────────────────────

/**
 * Compute the Hamming mean (majority-vote centroid) of a set of
 * braille lattice points.
 *
 * For each bit position, the centroid takes the majority value.
 * This is the discrete analogue of the arithmetic mean — the point
 * that minimizes total Hamming distance to all inputs.
 *
 * Crucially, this centroid SNAPS to a valid lattice point.
 * There is no "drift" — only discrete cell flips.
 */
export function brailleCentroid(points: BrailleLatticePoint[]): BrailleLatticePoint {
  if (points.length === 0) {
    throw new Error("Cannot compute centroid of empty set");
  }

  const n = points.length;
  const bitStrings = points.map(toBitString);
  const centroidBits: boolean[] = new Array(72);

  for (let i = 0; i < 72; i++) {
    let ones = 0;
    for (let j = 0; j < n; j++) {
      if (bitStrings[j][i]) ones++;
    }
    centroidBits[i] = ones > n / 2;
  }

  return fromBitString(centroidBits);
}

/**
 * Prestige-weighted Hamming mean.
 * Each point's vote is weighted by its prestige.
 */
export function weightedBrailleCentroid(
  points: BrailleLatticePoint[],
  weights: number[],
): BrailleLatticePoint {
  if (points.length === 0) {
    throw new Error("Cannot compute centroid of empty set");
  }

  const bitStrings = points.map(toBitString);
  const totalWeight = weights.reduce((a, b) => a + b, 0);
  const centroidBits: boolean[] = new Array(72);

  for (let i = 0; i < 72; i++) {
    let weightedOnes = 0;
    for (let j = 0; j < points.length; j++) {
      if (bitStrings[j][i]) weightedOnes += weights[j];
    }
    centroidBits[i] = weightedOnes > totalWeight / 2;
  }

  return fromBitString(centroidBits);
}

// ─── Cell Flip Analysis ──────────────────────────────────────────────

export interface CellFlip {
  axis: Axis;
  dotIndex: number;
  dotName: string;
  from: boolean;
  to: boolean;
  interpretation: string;
}

const DOT_NAMES = [
  "positive polarity",
  "negative polarity",
  "tension",
  "intensity-high",
  "intensity-low",
  "rigidity",
];

/**
 * Analyze the cell flips between two braille lattice points.
 * This is the discrete analogue of "centroid drift" — each flip
 * is a countable, interpretable event.
 */
export function analyzeCellFlips(
  before: BrailleLatticePoint,
  after: BrailleLatticePoint,
): CellFlip[] {
  const flips: CellFlip[] = [];

  for (const axis of AXES) {
    const cellBefore = before[axis];
    const cellAfter = after[axis];

    for (let d = 0; d < 6; d++) {
      if (cellBefore.dots[d] !== cellAfter.dots[d]) {
        flips.push({
          axis,
          dotIndex: d,
          dotName: DOT_NAMES[d],
          from: cellBefore.dots[d],
          to: cellAfter.dots[d],
          interpretation: interpretFlip(axis, d, cellBefore.dots[d], cellAfter.dots[d]),
        });
      }
    }
  }

  return flips;
}

function interpretFlip(axis: Axis, dotIndex: number, from: boolean, to: boolean): string {
  const dir = to ? "activated" : "deactivated";
  switch (dotIndex) {
    case 0:
      return `${axis}: positive polarity ${dir} — ${to ? "axis becomes salient" : "axis fades"}`;
    case 1:
      return `${axis}: negative polarity ${dir} — ${to ? "counter-axis rises" : "opposition weakens"}`;
    case 2:
      return `${axis}: tension ${dir} — ${to ? "doctrinal ambiguity emerges" : "ambiguity resolves"}`;
    case 3:
      return `${axis}: intensity shifts ${to ? "higher" : "lower"} — ${to ? "conviction strengthens" : "conviction weakens"}`;
    case 4:
      return `${axis}: intensity fine-tune ${dir}`;
    case 5:
      return `${axis}: rigidity ${dir} — ${to ? "doctrine crystallizes" : "doctrine becomes fluid"}`;
    default:
      return `${axis}: dot ${dotIndex} ${dir}`;
  }
}

// ─── Channel Invariance Test ─────────────────────────────────────────

export interface ChannelInvarianceResult {
  /** Continuous-space cosine similarity between centroids */
  continuousSimilarity: number;
  /** Braille-space normalized Hamming distance between centroids */
  brailleDistance: number;
  /** Number of cell flips between the two braille centroids */
  cellFlips: number;
  /** Whether the braille centroids are identical (perfect invariance) */
  isInvariant: boolean;
  /** Detailed flip analysis */
  flips: CellFlip[];
  /** Interpretation */
  interpretation: string;
}

/**
 * Test channel invariance: do centroids computed from different
 * sensory modalities converge to the same braille configuration?
 *
 * This is the empirical test of the Accessibility Corollary:
 * if the braille centroids match (up to a tolerance), the deity
 * is invariant to sensory modality.
 *
 * The braille lattice makes this test *sharper* than continuous
 * cosine similarity — because the lattice snaps to discrete
 * configurations, small continuous differences either vanish
 * (same cell) or become countable (cell flip).
 */
export function testChannelInvariance(
  unrestrictedCentroid: Vec,
  restrictedCentroid: Vec,
  tolerance: number = 0,
): ChannelInvarianceResult {
  // Continuous comparison
  let dotProd = 0, normA = 0, normB = 0;
  for (const a of AXES) {
    dotProd += unrestrictedCentroid[a] * restrictedCentroid[a];
    normA += unrestrictedCentroid[a] ** 2;
    normB += restrictedCentroid[a] ** 2;
  }
  const continuousSimilarity = dotProd / (Math.sqrt(normA) * Math.sqrt(normB) || 1);

  // Braille comparison
  const brailleA = encodeToBraille(unrestrictedCentroid);
  const brailleB = encodeToBraille(restrictedCentroid);
  const dist = hammingDistance(brailleA, brailleB);
  const flips = analyzeCellFlips(brailleA, brailleB);

  const isInvariant = dist <= tolerance;

  let interpretation: string;
  if (isInvariant) {
    interpretation = "Perfect channel invariance: the deity's braille configuration is identical across sensory modalities. The theological structure survives compression and is independent of the channel through which it was transmitted.";
  } else if (dist <= 3) {
    interpretation = `Near-invariance: ${dist} cell flip(s) between modalities. The core theological structure is preserved; differences are minor intensity or rigidity variations.`;
  } else if (dist <= 8) {
    interpretation = `Partial invariance: ${dist} cell flips. The deity's dominant structure is preserved but secondary axes show modality-dependent variation.`;
  } else {
    interpretation = `Significant divergence: ${dist} cell flips. The sensory restriction materially alters the emergent deity structure. The Accessibility Corollary may not hold for this restriction level.`;
  }

  return {
    continuousSimilarity,
    brailleDistance: dist / 72,
    cellFlips: dist,
    isInvariant,
    flips,
    interpretation,
  };
}

// ─── Braille Unicode Rendering ───────────────────────────────────────

/**
 * Convert a braille cell to its Unicode braille character.
 * Standard braille dot numbering:
 *   1 4     bit 0  bit 3
 *   2 5  →  bit 1  bit 4
 *   3 6     bit 2  bit 5
 *
 * Unicode braille starts at U+2800, with each dot adding:
 *   dot 1 = +1, dot 2 = +2, dot 3 = +4,
 *   dot 4 = +8, dot 5 = +16, dot 6 = +32
 */
export function cellToUnicode(cell: BrailleCell): string {
  let code = 0x2800;
  if (cell.dots[0]) code += 1;   // dot 1
  if (cell.dots[1]) code += 2;   // dot 2
  if (cell.dots[2]) code += 4;   // dot 3
  if (cell.dots[3]) code += 8;   // dot 4
  if (cell.dots[4]) code += 16;  // dot 5
  if (cell.dots[5]) code += 32;  // dot 6
  return String.fromCharCode(code);
}

/**
 * Render a full braille lattice point as a string of 12 braille characters.
 * Each character encodes one theological axis.
 */
export function latticeToUnicode(lattice: BrailleLatticePoint): string {
  return AXES.map((a) => cellToUnicode(lattice[a])).join("");
}

/**
 * Render a braille lattice point as a labeled string.
 */
export function latticeToLabeledString(lattice: BrailleLatticePoint): string {
  return AXES.map((a) => `${a}: ${cellToUnicode(lattice[a])}`).join("  ");
}

/**
 * Render a deity's braille signature: name + braille string.
 */
export function deityBrailleSignature(name: string, vec: Vec): string {
  const lattice = encodeToBraille(vec);
  return `${name}: ${latticeToUnicode(lattice)}`;
}

// ─── Phase Transition Detection via Braille ──────────────────────────

/**
 * In braille space, phase transitions manifest as sudden bursts of
 * cell flips. This function tracks the flip rate over time to detect
 * transitions.
 *
 * Returns the number of cell flips between consecutive centroid snapshots.
 * A spike in flip count indicates a phase transition.
 */
export function detectPhaseTransitions(
  centroidHistory: Vec[],
): { step: number; flips: number; totalFlips: number }[] {
  if (centroidHistory.length < 2) return [];

  const results: { step: number; flips: number; totalFlips: number }[] = [];
  let totalFlips = 0;

  for (let i = 1; i < centroidHistory.length; i++) {
    const before = encodeToBraille(centroidHistory[i - 1]);
    const after = encodeToBraille(centroidHistory[i]);
    const flips = hammingDistance(before, after);
    totalFlips += flips;
    results.push({ step: i, flips, totalFlips });
  }

  return results;
}

/**
 * Identify punctuated equilibrium: periods of stability (0 flips)
 * interrupted by bursts (many flips).
 */
export function identifyEquilibriumPeriods(
  flipHistory: { step: number; flips: number }[],
  stabilityThreshold: number = 0,
): { start: number; end: number; duration: number; type: "stable" | "transition" }[] {
  const periods: { start: number; end: number; duration: number; type: "stable" | "transition" }[] = [];
  let currentStart = 0;
  let currentType: "stable" | "transition" = flipHistory[0]?.flips <= stabilityThreshold ? "stable" : "transition";

  for (let i = 1; i < flipHistory.length; i++) {
    const isStable = flipHistory[i].flips <= stabilityThreshold;
    const newType = isStable ? "stable" : "transition";

    if (newType !== currentType) {
      periods.push({
        start: currentStart,
        end: i - 1,
        duration: i - currentStart,
        type: currentType,
      });
      currentStart = i;
      currentType = newType;
    }
  }

  // Close final period
  if (flipHistory.length > 0) {
    periods.push({
      start: currentStart,
      end: flipHistory.length - 1,
      duration: flipHistory.length - currentStart,
      type: currentType,
    });
  }

  return periods;
}
