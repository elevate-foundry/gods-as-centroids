/**
 * Historical Religious Data for Backtesting
 * ==========================================
 * Real-world estimates of religious diversity metrics across history.
 * Sources: Pew Research, World Religion Database, Stark (1996),
 * Norris & Inglehart (2004), Johnson & Grim (2013).
 *
 * N_eff = effective number of major traditions (analogous to centroid count)
 * dominance = fraction of world population in largest tradition
 * entropy = Shannon entropy of tradition distribution
 */

export interface HistoricalEpoch {
  year: number;
  label: string;
  nEff: number;
  dominance: number;
  entropy: number;
  majorTraditions: string[];
  event?: string;
  coercionEstimate: number; // 0-1 estimated coercion level
  color?: string;
}

export interface HistoricalEvent {
  year: number;
  label: string;
  type: "prophet" | "war" | "schism" | "syncretism" | "empire" | "reform" | "decline";
  description: string;
}

// ─── Epoch Data ──────────────────────────────────────────────────────
// Each epoch represents a snapshot of the global religious landscape.
// N_eff is estimated as the number of traditions each holding >3% of
// the relevant population. Dominance is the largest single tradition's share.

export const HISTORICAL_EPOCHS: HistoricalEpoch[] = [
  {
    year: -3000,
    label: "Early Bronze Age",
    nEff: 12,
    dominance: 0.15,
    entropy: 3.2,
    majorTraditions: ["Sumerian", "Egyptian", "Indus Valley", "Chinese folk", "Minoan", "Various animisms"],
    coercionEstimate: 0.05,
  },
  {
    year: -2000,
    label: "Middle Bronze Age",
    nEff: 10,
    dominance: 0.18,
    entropy: 3.0,
    majorTraditions: ["Egyptian", "Mesopotamian", "Vedic", "Chinese folk", "Canaanite"],
    event: "Rise of organized priesthoods",
    coercionEstimate: 0.10,
  },
  {
    year: -1500,
    label: "Late Bronze Age",
    nEff: 9,
    dominance: 0.20,
    entropy: 2.8,
    majorTraditions: ["Egyptian", "Mesopotamian", "Vedic", "Canaanite", "Hittite", "Chinese"],
    event: "Rigveda composed",
    coercionEstimate: 0.12,
  },
  {
    year: -1000,
    label: "Iron Age Begins",
    nEff: 8,
    dominance: 0.22,
    entropy: 2.7,
    majorTraditions: ["Vedic Hinduism", "Yahwism", "Zoroastrianism", "Greek polytheism", "Chinese folk"],
    event: "Mosaic distinction emerges",
    coercionEstimate: 0.18,
  },
  {
    year: -600,
    label: "Axial Age",
    nEff: 10,
    dominance: 0.18,
    entropy: 3.1,
    majorTraditions: ["Hinduism", "Buddhism", "Jainism", "Zoroastrianism", "Judaism", "Greek philosophy", "Confucianism", "Taoism"],
    event: "Buddha, Confucius, Zoroaster, Hebrew prophets",
    coercionEstimate: 0.10,
  },
  {
    year: -300,
    label: "Hellenistic Period",
    nEff: 9,
    dominance: 0.20,
    entropy: 2.9,
    majorTraditions: ["Hinduism", "Buddhism", "Greek syncretism", "Judaism", "Zoroastrianism", "Roman polytheism"],
    event: "Alexander's conquests drive syncretism",
    coercionEstimate: 0.20,
  },
  {
    year: 0,
    label: "Roman Empire",
    nEff: 8,
    dominance: 0.25,
    entropy: 2.6,
    majorTraditions: ["Roman polytheism", "Hinduism", "Buddhism", "Judaism", "Mystery cults", "Zoroastrianism"],
    event: "Birth of Christianity",
    coercionEstimate: 0.25,
  },
  {
    year: 100,
    label: "Early Christianity",
    nEff: 8,
    dominance: 0.25,
    entropy: 2.6,
    majorTraditions: ["Roman polytheism", "Hinduism", "Buddhism", "Early Christianity", "Judaism", "Zoroastrianism"],
    event: "Paul's missions; Christianity spreads",
    coercionEstimate: 0.22,
  },
  {
    year: 325,
    label: "Council of Nicaea",
    nEff: 6,
    dominance: 0.30,
    entropy: 2.3,
    majorTraditions: ["Christianity", "Roman polytheism", "Hinduism", "Buddhism", "Zoroastrianism", "Judaism"],
    event: "Constantine legalizes Christianity; Nicene Creed",
    coercionEstimate: 0.40,
  },
  {
    year: 400,
    label: "Theodosian Decree",
    nEff: 5,
    dominance: 0.35,
    entropy: 2.1,
    majorTraditions: ["Christianity", "Hinduism", "Buddhism", "Zoroastrianism", "Chinese folk"],
    event: "Christianity becomes Roman state religion; paganism banned",
    coercionEstimate: 0.60,
  },
  {
    year: 632,
    label: "Rise of Islam",
    nEff: 5,
    dominance: 0.32,
    entropy: 2.2,
    majorTraditions: ["Christianity", "Islam", "Hinduism", "Buddhism", "Chinese folk"],
    event: "Death of Muhammad; Islamic expansion begins",
    coercionEstimate: 0.45,
  },
  {
    year: 750,
    label: "Abbasid Golden Age",
    nEff: 5,
    dominance: 0.30,
    entropy: 2.2,
    majorTraditions: ["Christianity", "Islam", "Hinduism", "Buddhism", "Chinese folk"],
    event: "Islam reaches from Spain to Central Asia",
    coercionEstimate: 0.50,
  },
  {
    year: 1000,
    label: "Medieval Peak",
    nEff: 5,
    dominance: 0.33,
    entropy: 2.1,
    majorTraditions: ["Christianity", "Islam", "Hinduism", "Buddhism", "Chinese folk"],
    event: "Great Schism (East/West Christianity) 1054",
    coercionEstimate: 0.55,
  },
  {
    year: 1200,
    label: "Crusades Era",
    nEff: 5,
    dominance: 0.32,
    entropy: 2.1,
    majorTraditions: ["Christianity", "Islam", "Hinduism", "Buddhism", "Chinese folk"],
    event: "Crusades; Mongol expansion disrupts all traditions",
    coercionEstimate: 0.60,
  },
  {
    year: 1500,
    label: "Pre-Reformation",
    nEff: 5,
    dominance: 0.30,
    entropy: 2.2,
    majorTraditions: ["Catholic Christianity", "Islam", "Hinduism", "Buddhism", "Chinese folk"],
    event: "European colonialism begins",
    coercionEstimate: 0.55,
  },
  {
    year: 1520,
    label: "Protestant Reformation",
    nEff: 6,
    dominance: 0.28,
    entropy: 2.4,
    majorTraditions: ["Catholic", "Protestant", "Islam", "Hinduism", "Buddhism", "Chinese folk"],
    event: "Luther's 95 Theses; Christianity fractures",
    coercionEstimate: 0.50,
  },
  {
    year: 1650,
    label: "Wars of Religion",
    nEff: 6,
    dominance: 0.27,
    entropy: 2.4,
    majorTraditions: ["Catholic", "Protestant", "Islam", "Hinduism", "Buddhism", "Chinese folk"],
    event: "Thirty Years' War; Peace of Westphalia",
    coercionEstimate: 0.55,
  },
  {
    year: 1800,
    label: "Enlightenment",
    nEff: 6,
    dominance: 0.28,
    entropy: 2.3,
    majorTraditions: ["Catholic", "Protestant", "Islam", "Hinduism", "Buddhism", "Secular/Deist"],
    event: "Rise of secularism and religious freedom",
    coercionEstimate: 0.35,
  },
  {
    year: 1900,
    label: "Early Modern",
    nEff: 6,
    dominance: 0.33,
    entropy: 2.2,
    majorTraditions: ["Christianity", "Islam", "Hinduism", "Buddhism", "Chinese folk", "Secular"],
    event: "Colonial missions; global Christianity peaks",
    coercionEstimate: 0.30,
  },
  {
    year: 1950,
    label: "Post-WWII",
    nEff: 6,
    dominance: 0.33,
    entropy: 2.2,
    majorTraditions: ["Christianity", "Islam", "Hinduism", "Buddhism", "Chinese folk", "Secular/Atheist"],
    event: "Communist state atheism; decolonization",
    coercionEstimate: 0.35,
  },
  {
    year: 1970,
    label: "Secularization Wave",
    nEff: 7,
    dominance: 0.32,
    entropy: 2.3,
    majorTraditions: ["Christianity", "Islam", "Secular", "Hinduism", "Buddhism", "Chinese folk", "New Age"],
    event: "Western secularization accelerates",
    coercionEstimate: 0.20,
  },
  {
    year: 2000,
    label: "Turn of Millennium",
    nEff: 7,
    dominance: 0.31,
    entropy: 2.4,
    majorTraditions: ["Christianity", "Islam", "Secular/Unaffiliated", "Hinduism", "Buddhism", "Chinese folk", "Other"],
    event: "Internet era; information access transforms belief",
    coercionEstimate: 0.15,
  },
  {
    year: 2025,
    label: "Present Day",
    nEff: 7,
    dominance: 0.31,
    entropy: 2.4,
    majorTraditions: ["Christianity (31%)", "Islam (25%)", "Unaffiliated (16%)", "Hinduism (15%)", "Buddhism (7%)", "Folk (5%)", "Other (1%)"],
    event: "AI and digital religion emerge",
    coercionEstimate: 0.12,
  },
];

// ─── Key Historical Events ──────────────────────────────────────────
export const HISTORICAL_EVENTS: HistoricalEvent[] = [
  { year: -1350, label: "Akhenaten", type: "prophet", description: "First recorded monotheistic experiment (Atenism) — reversed after his death" },
  { year: -600, label: "Axial Age", type: "prophet", description: "Buddha, Confucius, Zoroaster, Hebrew prophets emerge independently" },
  { year: -330, label: "Alexander", type: "syncretism", description: "Hellenistic syncretism blends Greek, Egyptian, Persian traditions" },
  { year: 30, label: "Jesus", type: "prophet", description: "Christianity nucleates as a Jewish sect" },
  { year: 70, label: "Temple Destroyed", type: "war", description: "Roman destruction of Jerusalem Temple transforms Judaism" },
  { year: 313, label: "Edict of Milan", type: "empire", description: "Constantine legalizes Christianity" },
  { year: 380, label: "Theodosius", type: "empire", description: "Christianity becomes sole state religion; paganism criminalized" },
  { year: 622, label: "Muhammad", type: "prophet", description: "Hijra — Islam begins rapid expansion" },
  { year: 1054, label: "Great Schism", type: "schism", description: "Christianity splits into Catholic and Orthodox" },
  { year: 1095, label: "First Crusade", type: "war", description: "Military conflict reshapes Christian-Muslim boundaries" },
  { year: 1517, label: "Luther", type: "schism", description: "Protestant Reformation fractures Western Christianity" },
  { year: 1618, label: "30 Years' War", type: "war", description: "Deadliest European religious war; leads to secular state model" },
  { year: 1648, label: "Westphalia", type: "reform", description: "Peace of Westphalia establishes religious tolerance principle" },
  { year: 1789, label: "French Revolution", type: "decline", description: "Radical secularization; dechristianization campaign" },
  { year: 1830, label: "Joseph Smith", type: "prophet", description: "Mormonism founded — new prophetic tradition in modern era" },
  { year: 1844, label: "Bahá'u'lláh", type: "prophet", description: "Bahá'í Faith emerges from Shia Islam" },
  { year: 1917, label: "Russian Revolution", type: "decline", description: "State atheism imposed; massive forced secularization" },
  { year: 1947, label: "Partition", type: "schism", description: "India-Pakistan partition along religious lines" },
  { year: 1979, label: "Iranian Revolution", type: "empire", description: "Islamic theocracy established; global Islamist revival" },
  { year: 2001, label: "9/11", type: "war", description: "Religious conflict enters global consciousness" },
];

// ─── Future Projection Models ────────────────────────────────────────
export interface Projection {
  year: number;
  nEff: number;
  dominance: number;
  entropy: number;
  scenario: string;
}

/**
 * Generate future projections blending historical endpoint with
 * the simulation's current state. When the sim has run, its metrics
 * pull the projections toward the world the sim is modeling.
 *
 * simWeight ∈ [0,1] controls how much the sim influences projections
 * (0 = pure historical extrapolation, 1 = pure sim extrapolation).
 */
export function generateProjections(
  currentNEff: number,
  currentDominance: number,
  currentEntropy: number,
  currentCoercion: number,
  currentMutationRate: number,
  simNEff?: number,
  simDominance?: number,
  simEntropy?: number,
  simWeight: number = 0.4,
): { baseline: Projection[]; optimistic: Projection[]; pessimistic: Projection[] } {
  const years = [2030, 2040, 2050, 2060, 2075, 2100];

  // Blend historical endpoint with sim state
  const sw = simNEff != null ? simWeight : 0;
  const anchorNEff = currentNEff * (1 - sw) + (simNEff ?? currentNEff) * sw;
  const anchorDom = currentDominance * (1 - sw) + (simDominance ?? currentDominance) * sw;
  const anchorEnt = currentEntropy * (1 - sw) + (simEntropy ?? currentEntropy) * sw;

  // Baseline: extrapolate current trends
  // Pew Research projects Islam catching Christianity by ~2060
  // Unaffiliated growing in West, shrinking share globally
  const baseline = years.map((year) => {
    const dt = (year - 2025) / 75; // normalized time 0→1
    const nEff = anchorNEff + dt * (anchorNEff - 7) * 0.3;
    const dominance = anchorDom + dt * (anchorDom - 0.31) * 0.2 - 0.01 * dt;
    const entropy = anchorEnt + dt * (anchorEnt - 2.4) * 0.2 + 0.05 * dt;
    return {
      year,
      nEff: Math.max(1, Math.round(nEff * 10) / 10),
      dominance: Math.max(0.05, Math.min(1, dominance)),
      entropy: Math.max(0, entropy),
      scenario: "baseline",
    };
  });

  // Optimistic (pluralistic): low coercion, high mutation → more diversity
  const optimistic = years.map((year) => {
    const dt = (year - 2025) / 75;
    const diversityBoost = (1 - currentCoercion) * currentMutationRate * 10;
    const nEff = anchorNEff + dt * (2 + diversityBoost);
    const dominance = anchorDom - dt * (0.08 + diversityBoost * 0.02);
    const entropy = anchorEnt + dt * (0.4 + diversityBoost * 0.1);
    return {
      year,
      nEff: Math.max(1, Math.round(nEff * 10) / 10),
      dominance: Math.max(0.05, Math.min(1, dominance)),
      entropy: Math.max(0, entropy),
      scenario: "pluralistic",
    };
  });

  // Pessimistic (convergent): high coercion → fewer traditions
  const pessimistic = years.map((year) => {
    const dt = (year - 2025) / 75;
    const coercionEffect = currentCoercion * 5;
    const nEff = anchorNEff - dt * (1.5 + coercionEffect);
    const dominance = anchorDom + dt * (0.15 + coercionEffect * 0.05);
    const entropy = anchorEnt - dt * (0.5 + coercionEffect * 0.1);
    return {
      year,
      nEff: Math.max(1, Math.round(nEff * 10) / 10),
      dominance: Math.max(0.05, Math.min(1, dominance)),
      entropy: Math.max(0, entropy),
      scenario: "convergent",
    };
  });

  return { baseline, optimistic, pessimistic };
}

// ─── Map simulation time to historical time ──────────────────────────
// 1 sim step ≈ 1 year (adjustable), so step 0 = start of timeline
export function simStepToYear(step: number, startYear: number = -3000, yearsPerStep: number = 1): number {
  return startYear + step * yearsPerStep;
}

export function yearToSimStep(year: number, startYear: number = -3000, yearsPerStep: number = 1): number {
  return (year - startYear) / yearsPerStep;
}
