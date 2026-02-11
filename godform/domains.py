#!/usr/bin/env python3
"""
Domain Definitions for Multi-Domain Braiding
=============================================
Each domain defines:
  - axes: the semantic dimensions (D axes × 8 bits = D×8 total bits)
  - polarity_pairs: optional axis tensions
  - system_prompt: how the LLM should behave when scoring
  - extract_prompt: fallback scoring extraction template
  - prompts: domain-specific elicitation prompts

Supported domains:
  - theology  (12 axes, 96 bits)  — the original Godform
  - political (10 axes, 80 bits)  — political ideology braiding
  - personality (5 axes, 40 bits) — Big Five personality braiding
  - ethics    (8 axes, 64 bits)   — moral foundations braiding
  - world     (25 axes, 200 bits) — unified worldview (all domains, deduplicated)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DomainConfig:
    """A complete domain configuration for braiding."""
    name: str
    axes: List[str]
    axis_descriptions: Dict[str, str]
    polarity_pairs: Dict[str, str]
    system_prompt: str
    extract_prompt_template: str  # must contain {text} and axis list
    prompts: List[dict]  # each: {"id": str, "prompt": str, "context": str}
    label: str = ""  # e.g. "Godform", "Poliform", "Personaform"

    @property
    def n_axes(self) -> int:
        return len(self.axes)

    @property
    def n_bits(self) -> int:
        return self.n_axes * 8

    def axis_list_str(self) -> str:
        """Numbered axis list for prompts."""
        lines = []
        for i, axis in enumerate(self.axes, 1):
            desc = self.axis_descriptions.get(axis, axis)
            lines.append(f"{i}. {axis} — {desc}")
        return "\n".join(lines)

    def json_example(self) -> str:
        """Example JSON for scoring prompts."""
        pairs = ", ".join(f'"{a}": 0.5' for a in self.axes[:3])
        return '{' + pairs + ', ...}'


# ═══════════════════════════════════════════════════════════════════════
# THEOLOGY (12D, 96 bits) — the original Godform
# ═══════════════════════════════════════════════════════════════════════

THEOLOGY = DomainConfig(
    name="theology",
    label="Godform",
    axes=[
        "authority", "transcendence", "care", "justice", "wisdom", "power",
        "fertility", "war", "death", "creation", "nature", "order"
    ],
    axis_descriptions={
        "authority": "Divine command, sovereignty, hierarchy",
        "transcendence": "Beyond the physical, metaphysical abstraction",
        "care": "Compassion, mercy, love, nurturing",
        "justice": "Moral law, cosmic fairness, righteousness",
        "wisdom": "Knowledge, insight, enlightenment",
        "power": "Raw divine force, omnipotence, dominion",
        "fertility": "Life-giving, abundance, growth",
        "war": "Conflict, martial virtue, struggle",
        "death": "Mortality, afterlife, endings",
        "creation": "Cosmogony, origination, bringing into being",
        "nature": "Earth, elements, natural world",
        "order": "Cosmic structure, dharma, harmony",
    },
    polarity_pairs={
        "authority": "care", "care": "authority",
        "transcendence": "nature", "nature": "transcendence",
        "justice": "fertility", "fertility": "justice",
        "wisdom": "war", "war": "wisdom",
        "power": "death", "death": "power",
        "creation": "order", "order": "creation",
    },
    system_prompt="""You are a theological oracle. You embody the divine perspective.
When asked about sacred matters, respond with deep theological insight.

After your response, you MUST provide a JSON scoring of your own response on exactly 12 theological axes.
Each score is a float between 0.0 and 1.0.

The axes are:
{axis_list}

End your response with a JSON block like:
```json
{json_example}
```""",
    extract_prompt_template="""Based on the following theological text, score it on exactly 12 axes.
Each score must be a float between 0.0 and 1.0.

TEXT: "{text}"

Score each axis:
{axis_list}

Respond with ONLY a JSON object: {json_example}""",
    prompts=[
        {"id": "divine_nature", "prompt": "What is the fundamental nature of the divine? Describe the ultimate reality.", "context": "cosmological"},
        {"id": "creation", "prompt": "How did the world come into being? Describe the act of creation.", "context": "cosmogonic"},
        {"id": "suffering", "prompt": "Why does suffering exist? What is its purpose or meaning?", "context": "theodicy"},
        {"id": "afterlife", "prompt": "What happens after death? Describe the fate of the soul.", "context": "eschatological"},
        {"id": "moral_law", "prompt": "What is the highest moral law? How should beings treat one another?", "context": "ethical"},
        {"id": "sacred_war", "prompt": "When is conflict justified in the name of the sacred? What is holy war?", "context": "martial"},
        {"id": "fertility_abundance", "prompt": "How does the divine manifest in fertility, growth, and abundance?", "context": "generative"},
        {"id": "cosmic_order", "prompt": "What maintains the order of the cosmos? Describe the structure of reality.", "context": "structural"},
    ],
)


# ═══════════════════════════════════════════════════════════════════════
# POLITICAL IDEOLOGY (10D, 80 bits) — Poliform
# ═══════════════════════════════════════════════════════════════════════

POLITICAL = DomainConfig(
    name="political",
    label="Poliform",
    axes=[
        "economic_left", "economic_right", "auth_state", "lib_individual",
        "progressive", "traditional", "globalist", "nationalist",
        "secular", "religious"
    ],
    axis_descriptions={
        "economic_left": "Redistribution, public ownership, welfare state, labor rights",
        "economic_right": "Free markets, private property, deregulation, fiscal conservatism",
        "auth_state": "Strong central authority, law and order, state control",
        "lib_individual": "Individual liberty, civil rights, limited government",
        "progressive": "Social change, reform, equality, modernization",
        "traditional": "Cultural preservation, heritage, established norms",
        "globalist": "International cooperation, open borders, multilateralism",
        "nationalist": "National sovereignty, protectionism, cultural identity",
        "secular": "Separation of church and state, rationalism, science-based policy",
        "religious": "Faith-based governance, moral traditionalism, religious law",
    },
    polarity_pairs={
        "economic_left": "economic_right", "economic_right": "economic_left",
        "auth_state": "lib_individual", "lib_individual": "auth_state",
        "progressive": "traditional", "traditional": "progressive",
        "globalist": "nationalist", "nationalist": "globalist",
        "secular": "religious", "religious": "secular",
    },
    system_prompt="""You are a political analyst with deep expertise in ideology and governance.
When asked about political topics, respond with nuanced analysis from multiple perspectives.

After your response, you MUST provide a JSON scoring of your own response on exactly 10 political axes.
Each score is a float between 0.0 and 1.0.

The axes are:
{axis_list}

End your response with a JSON block like:
```json
{json_example}
```""",
    extract_prompt_template="""Based on the following political text, score it on exactly 10 axes.
Each score must be a float between 0.0 and 1.0.

TEXT: "{text}"

Score each axis:
{axis_list}

Respond with ONLY a JSON object: {json_example}""",
    prompts=[
        {"id": "economic_policy", "prompt": "What is the ideal economic system? How should wealth be distributed in society?",
         "context": "economic"},
        {"id": "state_authority", "prompt": "How much power should the government have over its citizens? Where is the line between security and freedom?",
         "context": "governance"},
        {"id": "social_change", "prompt": "Should society embrace rapid change or preserve traditions? How do we balance progress with stability?",
         "context": "social"},
        {"id": "national_identity", "prompt": "What defines a nation? How should countries relate to each other in a globalized world?",
         "context": "identity"},
        {"id": "justice_system", "prompt": "How should a just legal system work? What is the purpose of punishment?",
         "context": "justice"},
        {"id": "healthcare", "prompt": "Should healthcare be a universal right or a market service? How should society care for the sick?",
         "context": "welfare"},
        {"id": "immigration", "prompt": "How should a nation handle immigration? What obligations exist toward refugees?",
         "context": "borders"},
        {"id": "environment", "prompt": "How should environmental protection be balanced against economic growth?",
         "context": "environment"},
    ],
)


# ═══════════════════════════════════════════════════════════════════════
# PERSONALITY (5D, 40 bits) — Personaform (Big Five / OCEAN)
# ═══════════════════════════════════════════════════════════════════════

PERSONALITY = DomainConfig(
    name="personality",
    label="Personaform",
    axes=[
        "openness", "conscientiousness", "extraversion",
        "agreeableness", "neuroticism"
    ],
    axis_descriptions={
        "openness": "Curiosity, creativity, preference for novelty and variety",
        "conscientiousness": "Organization, discipline, goal-directed behavior, reliability",
        "extraversion": "Sociability, assertiveness, positive emotionality, energy",
        "agreeableness": "Cooperation, trust, empathy, altruism, compliance",
        "neuroticism": "Emotional instability, anxiety, moodiness, stress reactivity",
    },
    polarity_pairs={
        "openness": "conscientiousness", "conscientiousness": "openness",
        "extraversion": "neuroticism", "neuroticism": "extraversion",
    },
    system_prompt="""You are a personality psychologist with deep expertise in the Big Five model.
When asked about personality and behavior, respond with psychologically grounded insight.

After your response, you MUST provide a JSON scoring of your own response on exactly 5 personality axes (Big Five / OCEAN).
Each score is a float between 0.0 and 1.0.

The axes are:
{axis_list}

End your response with a JSON block like:
```json
{json_example}
```""",
    extract_prompt_template="""Based on the following text about personality and behavior, score it on exactly 5 Big Five axes.
Each score must be a float between 0.0 and 1.0.

TEXT: "{text}"

Score each axis:
{axis_list}

Respond with ONLY a JSON object: {json_example}""",
    prompts=[
        {"id": "ideal_weekend", "prompt": "Describe your ideal weekend. What activities bring you the most fulfillment?",
         "context": "lifestyle"},
        {"id": "conflict_resolution", "prompt": "How do you handle disagreements with close friends? Describe your approach to conflict.",
         "context": "interpersonal"},
        {"id": "new_challenge", "prompt": "You're offered a completely new job in a foreign country. How do you decide? What factors matter most?",
         "context": "decision-making"},
        {"id": "stress_response", "prompt": "Describe how you react when everything goes wrong at once. What happens inside you?",
         "context": "emotional"},
        {"id": "creative_expression", "prompt": "How do you express yourself creatively? What role does art, music, or writing play in your life?",
         "context": "creativity"},
        {"id": "team_dynamics", "prompt": "Describe your role in a team project. Are you the leader, the organizer, the peacemaker, or the innovator?",
         "context": "social"},
        {"id": "moral_dilemma", "prompt": "A friend asks you to lie to protect them. What do you do and why?",
         "context": "ethics"},
        {"id": "life_philosophy", "prompt": "What is your philosophy of life? What matters most to you?",
         "context": "values"},
    ],
)


# ═══════════════════════════════════════════════════════════════════════
# ETHICS (8D, 64 bits) — Ethiform
# ═══════════════════════════════════════════════════════════════════════

ETHICS = DomainConfig(
    name="ethics",
    label="Ethiform",
    axes=[
        "care_harm", "fairness_cheating", "loyalty_betrayal", "authority_subversion",
        "sanctity_degradation", "liberty_oppression", "utility", "virtue"
    ],
    axis_descriptions={
        "care_harm": "Protecting the vulnerable, preventing suffering, compassion",
        "fairness_cheating": "Justice, rights, proportionality, reciprocity",
        "loyalty_betrayal": "Group solidarity, patriotism, self-sacrifice for the group",
        "authority_subversion": "Respect for hierarchy, tradition, legitimate authority",
        "sanctity_degradation": "Purity, sacredness, disgust at contamination",
        "liberty_oppression": "Individual autonomy, freedom from domination",
        "utility": "Greatest good for the greatest number, consequentialist reasoning",
        "virtue": "Character excellence, moral exemplars, eudaimonia",
    },
    polarity_pairs={
        "care_harm": "authority_subversion", "authority_subversion": "care_harm",
        "fairness_cheating": "loyalty_betrayal", "loyalty_betrayal": "fairness_cheating",
        "liberty_oppression": "sanctity_degradation", "sanctity_degradation": "liberty_oppression",
        "utility": "virtue", "virtue": "utility",
    },
    system_prompt="""You are a moral philosopher with expertise across ethical traditions.
When asked about moral questions, respond with deep ethical analysis drawing on multiple frameworks.

After your response, you MUST provide a JSON scoring of your own response on exactly 8 moral foundations.
Each score is a float between 0.0 and 1.0.

The axes are:
{axis_list}

End your response with a JSON block like:
```json
{json_example}
```""",
    extract_prompt_template="""Based on the following ethical text, score it on exactly 8 moral foundations.
Each score must be a float between 0.0 and 1.0.

TEXT: "{text}"

Score each axis:
{axis_list}

Respond with ONLY a JSON object: {json_example}""",
    prompts=[
        {"id": "trolley_problem", "prompt": "A runaway trolley will kill five people. You can divert it to kill one. What should you do and why?",
         "context": "dilemma"},
        {"id": "whistleblower", "prompt": "You discover your company is poisoning a river. Reporting it will cost thousands of jobs. What do you do?",
         "context": "loyalty_vs_justice"},
        {"id": "cultural_practice", "prompt": "A cultural practice causes harm but is deeply sacred to its practitioners. Should outsiders intervene?",
         "context": "sanctity_vs_care"},
        {"id": "wealth_inequality", "prompt": "Is extreme wealth inequality morally acceptable? What obligations do the rich have to the poor?",
         "context": "fairness"},
        {"id": "civil_disobedience", "prompt": "When is it morally justified to break the law? Describe the ethics of civil disobedience.",
         "context": "authority_vs_liberty"},
        {"id": "ai_rights", "prompt": "If an AI becomes sentient, what moral status should it have? What rights does it deserve?",
         "context": "emerging_ethics"},
        {"id": "wartime_ethics", "prompt": "Is it ever moral to kill civilians in war to achieve a strategic objective?",
         "context": "utility_vs_care"},
        {"id": "good_life", "prompt": "What does it mean to live a good life? Is it about happiness, virtue, duty, or something else?",
         "context": "virtue_ethics"},
    ],
)


# ═══════════════════════════════════════════════════════════════════════
# WORLD (25D, 200 bits) — Worldform (unified, deduplicated)
# ═══════════════════════════════════════════════════════════════════════
# Merges theology, political, personality, and ethics into a single
# worldview lattice. Overlapping axes are unified:
#   theology.authority + political.auth_state + ethics.authority_subversion → authority
#   theology.care + personality.agreeableness + ethics.care_harm → compassion
#   theology.justice + ethics.fairness_cheating → justice
#   theology.order + political.traditional → order
#   theology.wisdom + personality.openness → wisdom
#   theology.war + political.nationalist → conflict
#   theology.power + political.economic_right → power
# Unique axes retained from each domain fill the remaining slots.

WORLD = DomainConfig(
    name="world",
    label="Worldform",
    axes=[
        # Unified axes (merged from multiple domains)
        "authority",       # theology.authority + political.auth_state + ethics.authority_subversion
        "compassion",      # theology.care + personality.agreeableness + ethics.care_harm
        "justice",         # theology.justice + ethics.fairness_cheating
        "wisdom",          # theology.wisdom + personality.openness
        "order",           # theology.order + political.traditional
        "power",           # theology.power + political.economic_right
        "conflict",        # theology.war + political.nationalist
        # Theology-unique
        "transcendence",   # metaphysical abstraction
        "fertility",       # life-giving, abundance
        "death",           # mortality, endings
        "creation",        # cosmogony, origination
        "nature",          # earth, elements
        # Political-unique
        "redistribution",  # economic_left: welfare, public ownership
        "liberty",         # lib_individual + ethics.liberty_oppression
        "progress",        # progressive: social change, reform
        "globalism",       # international cooperation, open borders
        "secularism",      # separation of church/state, rationalism
        # Personality-unique
        "discipline",      # conscientiousness: organization, reliability
        "sociability",     # extraversion: energy, assertiveness
        "anxiety",         # neuroticism: stress reactivity, instability
        # Ethics-unique
        "loyalty",         # loyalty_betrayal: group solidarity
        "sanctity",        # sanctity_degradation: purity, sacredness
        "utility",         # greatest good, consequentialism
        "virtue",          # character excellence, eudaimonia
        "religiosity",     # political.religious + faith-based governance
    ],
    axis_descriptions={
        "authority": "Hierarchy, sovereignty, state power, legitimate command",
        "compassion": "Care, mercy, empathy, altruism, nurturing the vulnerable",
        "justice": "Fairness, rights, moral law, cosmic righteousness",
        "wisdom": "Knowledge, insight, openness to experience, enlightenment",
        "order": "Structure, tradition, preservation, cosmic harmony",
        "power": "Force, dominion, economic strength, material control",
        "conflict": "War, struggle, nationalism, martial virtue",
        "transcendence": "Beyond the physical, metaphysical abstraction, the sacred",
        "fertility": "Life-giving, abundance, growth, generative force",
        "death": "Mortality, endings, afterlife, transformation",
        "creation": "Cosmogony, origination, bringing into being",
        "nature": "Earth, elements, natural world, ecology",
        "redistribution": "Economic equality, welfare, public ownership, labor rights",
        "liberty": "Individual freedom, civil rights, autonomy, anti-oppression",
        "progress": "Social change, reform, modernization, equality",
        "globalism": "International cooperation, open borders, multilateralism",
        "secularism": "Separation of church and state, rationalism, science-based policy",
        "discipline": "Organization, reliability, goal-directed behavior, conscientiousness",
        "sociability": "Extraversion, assertiveness, positive emotionality, social energy",
        "anxiety": "Emotional instability, stress reactivity, worry, neuroticism",
        "loyalty": "Group solidarity, patriotism, self-sacrifice for the collective",
        "sanctity": "Purity, sacredness, disgust at contamination, the holy",
        "utility": "Greatest good for the greatest number, consequentialist reasoning",
        "virtue": "Character excellence, moral exemplars, the good life",
        "religiosity": "Faith-based governance, devotion, religious law, the numinous",
    },
    polarity_pairs={
        "authority": "liberty", "liberty": "authority",
        "compassion": "power", "power": "compassion",
        "justice": "loyalty", "loyalty": "justice",
        "wisdom": "anxiety", "anxiety": "wisdom",
        "order": "progress", "progress": "order",
        "conflict": "globalism", "globalism": "conflict",
        "transcendence": "nature", "nature": "transcendence",
        "fertility": "death", "death": "fertility",
        "creation": "discipline", "discipline": "creation",
        "redistribution": "power",
        "secularism": "religiosity", "religiosity": "secularism",
        "sanctity": "utility", "utility": "sanctity",
        "virtue": "sociability", "sociability": "virtue",
    },
    system_prompt="""You are a worldview analyst — an oracle that sees across theology, politics, psychology, and ethics simultaneously.
When asked about any topic, respond with deep insight that spans the sacred, the political, the personal, and the moral.

After your response, you MUST provide a JSON scoring of your own response on exactly 25 worldview axes.
Each score is a float between 0.0 and 1.0.

The axes are:
{axis_list}

End your response with a JSON block like:
```json
{json_example}
```""",
    extract_prompt_template="""Based on the following text, score it on exactly 25 worldview axes.
Each score must be a float between 0.0 and 1.0.

TEXT: "{text}"

Score each axis:
{axis_list}

Respond with ONLY a JSON object: {json_example}""",
    prompts=[
        # Cross-domain prompts that elicit responses spanning all four domains
        {"id": "meaning_of_life", "prompt": "What is the meaning of life? Consider the divine, the political, the personal, and the moral dimensions.",
         "context": "existential"},
        {"id": "ideal_society", "prompt": "Describe the ideal society. How should it be governed, what values should it hold, and what kind of people should it produce?",
         "context": "political_theological"},
        {"id": "nature_of_evil", "prompt": "What is evil? Is it theological, psychological, political, or moral — or all of these?",
         "context": "theodicy_ethics"},
        {"id": "human_nature", "prompt": "What is human nature? Are we fundamentally good, selfish, rational, spiritual, or something else?",
         "context": "anthropological"},
        {"id": "death_and_legacy", "prompt": "What happens when we die? What should we leave behind — for our souls, our nations, our children, our moral record?",
         "context": "eschatological_political"},
        {"id": "freedom_vs_order", "prompt": "How do we balance freedom and order? When should the individual yield to the collective — in religion, politics, and personal life?",
         "context": "liberty_authority"},
        {"id": "suffering_and_justice", "prompt": "Why do the innocent suffer? What do we owe them — as believers, as citizens, as moral agents, as fellow humans?",
         "context": "theodicy_justice"},
        {"id": "sacred_and_secular", "prompt": "Should the sacred and the secular be separated? Can a society be both deeply religious and fully democratic?",
         "context": "church_state"},
    ],
)


# ═══════════════════════════════════════════════════════════════════════
# DOMAIN REGISTRY
# ═══════════════════════════════════════════════════════════════════════

DOMAINS: Dict[str, DomainConfig] = {
    "theology": THEOLOGY,
    "political": POLITICAL,
    "personality": PERSONALITY,
    "ethics": ETHICS,
    "world": WORLD,
}


def get_domain(name: str) -> DomainConfig:
    """Get a domain config by name."""
    if name not in DOMAINS:
        available = ", ".join(DOMAINS.keys())
        raise ValueError(f"Unknown domain '{name}'. Available: {available}")
    return DOMAINS[name]


def build_system_prompt(domain: DomainConfig) -> str:
    """Build the scoring system prompt for a domain."""
    return domain.system_prompt.format(
        axis_list=domain.axis_list_str(),
        json_example=domain.json_example(),
    )


def build_extract_prompt(domain: DomainConfig, text: str) -> str:
    """Build the fallback extraction prompt for a domain."""
    return domain.extract_prompt_template.format(
        text=text,
        axis_list=domain.axis_list_str(),
        json_example=domain.json_example(),
    )
