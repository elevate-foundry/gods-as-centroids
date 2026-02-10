import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

const anthropic = process.env.ANTHROPIC_API_KEY
  ? new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
  : null;

interface DeityRequest {
  action: "describe" | "myth" | "prayer" | "doctrine" | "compare" | "fuse_describe";
  deity?: {
    name: string;
    vector: Record<string, number>;
    dominantAxes: string[];
    recessiveAxes: string[];
    source: string;
    adherents: number;
    profile: {
      archetype: string;
      domains: string[];
      worshipStyle: string;
      moralEmphasis: string;
      cosmologicalRole: string;
      eschatology: string;
      natureRelation: string;
      authorityStructure: string;
    };
    brailleSignature?: string;
    parentNames?: string[];
    mixingWeights?: number[];
  };
  deityB?: {
    name: string;
    vector: Record<string, number>;
    dominantAxes: string[];
    profile: {
      archetype: string;
      domains: string[];
    };
  };
}

function buildPrompt(req: DeityRequest): string {
  const d = req.deity!;
  const axes = Object.entries(d.vector)
    .sort(([, a], [, b]) => b - a)
    .map(([k, v]) => `  ${k}: ${(v as number).toFixed(3)}`)
    .join("\n");

  const base = `You are a theology engine — a system that generates structurally valid theological descriptions from mathematical deity-centroid vectors. You are NOT creating real religion. You are generating plausible theological structures that COULD have emerged in human history, based on the geometric properties of a 12-dimensional belief space.

DEITY: ${d.name}
SOURCE: ${d.source}${d.parentNames ? `\nPARENTS: ${d.parentNames.join(" + ")}` : ""}${d.mixingWeights ? `\nMIXING WEIGHTS: ${d.mixingWeights.map(w => w.toFixed(2)).join(", ")}` : ""}${d.brailleSignature ? `\nBRAILLE SIGNATURE: ${d.brailleSignature}` : ""}
ADHERENTS: ${d.adherents}

VECTOR (12D theological space):
${axes}

DOMINANT AXES: ${d.dominantAxes.join(", ")}
RECESSIVE AXES: ${d.recessiveAxes.join(", ")}

PROFILE:
  Archetype: ${d.profile.archetype}
  Domains: ${d.profile.domains.join(", ")}
  Worship: ${d.profile.worshipStyle}
  Moral emphasis: ${d.profile.moralEmphasis}
  Cosmological role: ${d.profile.cosmologicalRole}
  Eschatology: ${d.profile.eschatology}
  Nature relation: ${d.profile.natureRelation}
  Authority: ${d.profile.authorityStructure}`;

  switch (req.action) {
    case "describe":
      return `${base}

Generate a rich theological description in 3-4 paragraphs. Cover:
1. The deity's essential nature and character
2. How worshippers experience and relate to this deity
3. The social/political structures this deity implies
4. What makes this deity distinctive in the landscape of possible gods

Write in an academic but evocative style. This is structural analysis, not devotion.`;

    case "myth":
      return `${base}

Generate a creation myth fragment (200-300 words) that this deity's followers might tell. The myth should:
- Reflect the dominant axes (${d.dominantAxes.join(", ")})
- Explain why the recessive axes (${d.recessiveAxes.join(", ")}) are absent or diminished
- Feel like it could be a real myth from human history
- Have literary quality

Write it as narrative, not analysis.`;

    case "prayer":
      return `${base}

Generate a prayer or invocation (100-150 words) that a follower of this deity might speak. It should:
- Address the deity by name or title consistent with the archetype
- Reflect the moral emphasis and worship style
- Feel authentic — like something from a real liturgical tradition
- Be poetic but not overwrought

Write only the prayer text, no commentary.`;

    case "doctrine":
      return `${base}

Generate 5 core doctrinal tenets for this deity's tradition. Each should be:
- A single declarative sentence
- Grounded in the vector's dominant axes
- Internally consistent with the other tenets
- Distinguishable from generic monotheistic platitudes

Format as a numbered list. After the list, add one paragraph explaining the theological logic that connects them.`;

    case "compare": {
      const b = req.deityB!;
      const bAxes = Object.entries(b.vector)
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .map(([k, v]) => `  ${k}: ${(v as number).toFixed(3)}`)
        .join("\n");

      return `${base}

COMPARISON DEITY: ${b.name}
VECTOR:
${bAxes}
DOMINANT AXES: ${b.dominantAxes.join(", ")}
ARCHETYPE: ${b.profile.archetype}

Compare these two deities in 3-4 paragraphs:
1. Where they converge (shared high axes)
2. Where they diverge (opposing emphases)
3. What a syncretism between them would look like
4. Why historical contact between their traditions would produce tension on specific axes

Be precise about which axes drive the similarities and differences.`;
    }

    case "fuse_describe":
      return `${base}

This deity is a FUSION — a meta-centroid computed as a weighted average of multiple deity vectors. It is a second-order semantic attractor, not a historical deity.

Generate a description (3-4 paragraphs) of this novel theological structure:
1. What kind of god-concept emerges from this fusion?
2. What internal tensions exist (from the parent traditions)?
3. What kind of civilization might worship this deity?
4. How does this deity differ from any single parent tradition?

This is speculative theology-as-dynamical-systems. Be rigorous but imaginative.`;

    default:
      return `${base}\n\nDescribe this deity in 2-3 paragraphs.`;
  }
}

export async function POST(request: NextRequest) {
  try {
    const body: DeityRequest = await request.json();

    if (!body.deity) {
      return NextResponse.json({ error: "No deity provided" }, { status: 400 });
    }

    // If no API key, return the deterministic profile as fallback
    if (!anthropic) {
      const profile = body.deity.profile;
      const fallback = `**${profile.archetype}**\n\n` +
        `A deity of ${profile.domains.join(", ")}. ` +
        `Worshipped through ${profile.worshipStyle}. ` +
        `The moral framework emphasizes ${profile.moralEmphasis}. ` +
        `Cosmologically, this deity serves as ${profile.cosmologicalRole}. ` +
        `The eschatological vision: ${profile.eschatology}. ` +
        `Relationship to nature: ${profile.natureRelation}. ` +
        `The implied authority structure: ${profile.authorityStructure}.\n\n` +
        `_[Set ANTHROPIC_API_KEY in .env.local for richer LLM-generated descriptions]_`;

      return NextResponse.json({ text: fallback, source: "deterministic" });
    }

    const prompt = buildPrompt(body);

    const message = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    });

    const text = message.content
      .filter((b) => b.type === "text")
      .map((b) => (b as { type: "text"; text: string }).text)
      .join("\n\n");

    return NextResponse.json({ text, source: "anthropic" });
  } catch (error) {
    console.error("Theology API error:", error);
    return NextResponse.json(
      { error: "Failed to generate theology", details: String(error) },
      { status: 500 },
    );
  }
}
