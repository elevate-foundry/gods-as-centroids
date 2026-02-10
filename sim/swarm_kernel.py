from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Callable, Iterable
import math
import random
import statistics
from collections import Counter, defaultdict, deque
import itertools
import json

# ---------------------------- Utilities ---------------------------- #

AXES = [
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
]

# Sensory channel mappings for Accessibility Corollary
SENSORY_CHANNELS = {
    "sight": ["transcendence", "creation", "nature", "order"],
    "sound": ["authority", "wisdom", "war", "justice"], 
    "touch": ["care", "fertility", "power", "death"],
    "proprioception": ["authority", "power", "war", "order"],
    "smell": ["fertility", "nature", "death", "creation"],
    "taste": ["care", "wisdom", "justice", "transcendence"]
}

THEONYMS: Tuple[str, ...] = (
    # a small, diverse seed; purely for simulation; no theological claims
    "isis", "thor", "indra", "yah", "nana", "osiris", "zeus", "odin",
    "ra", "anu", "ishtar", "baal", "amun", "enar", "shen", "kami",
    "ahura", "taranis", "freya", "nut", "ptah", "apollo", "gaia",
    "hades", "marduk", "teotl", "manitou", "perun", "shango",
)

# Deity semantic priors - mapped to new AXES dimensions
DEITY_PRIORS: Dict[str, Dict[str, float]] = {
    "zeus": {"authority": 0.9, "transcendence": 0.8, "care": 0.3, "justice": 0.7, "wisdom": 0.6, "power": 0.9, "fertility": 0.2, "war": 0.8, "death": 0.1, "creation": 0.4, "nature": 0.3, "order": 0.8},
    "odin": {"authority": 0.8, "transcendence": 0.7, "care": 0.4, "justice": 0.6, "wisdom": 0.9, "power": 0.7, "fertility": 0.1, "war": 0.9, "death": 0.8, "creation": 0.3, "nature": 0.2, "order": 0.5},
    "amun": {"authority": 0.9, "transcendence": 0.9, "care": 0.6, "justice": 0.8, "wisdom": 0.8, "power": 0.8, "fertility": 0.3, "war": 0.2, "death": 0.1, "creation": 0.9, "nature": 0.1, "order": 0.9},
    "marduk": {"authority": 0.9, "transcendence": 0.6, "care": 0.5, "justice": 0.9, "wisdom": 0.7, "power": 0.9, "fertility": 0.1, "war": 0.8, "death": 0.3, "creation": 0.7, "nature": 0.1, "order": 0.9},
    "indra": {"authority": 0.8, "transcendence": 0.5, "care": 0.4, "justice": 0.7, "wisdom": 0.6, "power": 0.9, "fertility": 0.2, "war": 0.9, "death": 0.2, "creation": 0.3, "nature": 0.4, "order": 0.6},
    "shango": {"authority": 0.7, "transcendence": 0.4, "care": 0.3, "justice": 0.8, "wisdom": 0.5, "power": 0.8, "fertility": 0.1, "war": 0.7, "death": 0.2, "creation": 0.2, "nature": 0.6, "order": 0.5},
    "kami": {"authority": 0.3, "transcendence": 0.8, "care": 0.8, "justice": 0.4, "wisdom": 0.7, "power": 0.4, "fertility": 0.6, "war": 0.1, "death": 0.1, "creation": 0.5, "nature": 0.9, "order": 0.8},
    "manitou": {"authority": 0.2, "transcendence": 0.9, "care": 0.9, "justice": 0.3, "wisdom": 0.8, "power": 0.3, "fertility": 0.7, "war": 0.1, "death": 0.2, "creation": 0.6, "nature": 0.9, "order": 0.4},
    "apollo": {"authority": 0.6, "transcendence": 0.7, "care": 0.5, "justice": 0.6, "wisdom": 0.8, "power": 0.6, "fertility": 0.3, "war": 0.3, "death": 0.2, "creation": 0.7, "nature": 0.4, "order": 0.7},
    "freya": {"authority": 0.4, "transcendence": 0.5, "care": 0.8, "justice": 0.4, "wisdom": 0.6, "power": 0.5, "fertility": 0.9, "war": 0.6, "death": 0.4, "creation": 0.6, "nature": 0.7, "order": 0.3},
    "ptah": {"authority": 0.5, "transcendence": 0.4, "care": 0.6, "justice": 0.5, "wisdom": 0.7, "power": 0.6, "fertility": 0.2, "war": 0.1, "death": 0.1, "creation": 0.9, "nature": 0.2, "order": 0.8},
    "isis": {"authority": 0.4, "transcendence": 0.6, "care": 0.9, "justice": 0.6, "wisdom": 0.8, "power": 0.5, "fertility": 0.8, "war": 0.1, "death": 0.4, "creation": 0.7, "nature": 0.5, "order": 0.6},
    "ra": {"authority": 0.8, "transcendence": 0.8, "care": 0.5, "justice": 0.7, "wisdom": 0.7, "power": 0.8, "fertility": 0.3, "war": 0.4, "death": 0.2, "creation": 0.8, "nature": 0.6, "order": 0.8},
    "quetzal": {"authority": 0.6, "transcendence": 0.9, "care": 0.7, "justice": 0.6, "wisdom": 0.9, "power": 0.6, "fertility": 0.5, "war": 0.3, "death": 0.3, "creation": 0.8, "nature": 0.8, "order": 0.7},
    "tyr": {"authority": 0.7, "transcendence": 0.4, "care": 0.3, "justice": 0.9, "wisdom": 0.6, "power": 0.7, "fertility": 0.1, "war": 0.9, "death": 0.5, "creation": 0.2, "nature": 0.2, "order": 0.8},
    "bast": {"authority": 0.4, "transcendence": 0.5, "care": 0.8, "justice": 0.5, "wisdom": 0.6, "power": 0.5, "fertility": 0.7, "war": 0.3, "death": 0.2, "creation": 0.4, "nature": 0.7, "order": 0.6},
    "lugh": {"authority": 0.6, "transcendence": 0.6, "care": 0.5, "justice": 0.7, "wisdom": 0.8, "power": 0.7, "fertility": 0.4, "war": 0.6, "death": 0.2, "creation": 0.7, "nature": 0.5, "order": 0.6},
    "brigid": {"authority": 0.4, "transcendence": 0.6, "care": 0.9, "justice": 0.5, "wisdom": 0.8, "power": 0.5, "fertility": 0.8, "war": 0.2, "death": 0.1, "creation": 0.8, "nature": 0.7, "order": 0.6},
    "taranis": {"authority": 0.6, "transcendence": 0.5, "care": 0.3, "justice": 0.6, "wisdom": 0.5, "power": 0.8, "fertility": 0.2, "war": 0.7, "death": 0.3, "creation": 0.4, "nature": 0.6, "order": 0.5},
    "nana": {"authority": 0.3, "transcendence": 0.7, "care": 0.9, "justice": 0.4, "wisdom": 0.7, "power": 0.4, "fertility": 0.9, "war": 0.1, "death": 0.2, "creation": 0.6, "nature": 0.8, "order": 0.5},
    "enar": {"authority": 0.5, "transcendence": 0.6, "care": 0.6, "justice": 0.5, "wisdom": 0.6, "power": 0.5, "fertility": 0.4, "war": 0.3, "death": 0.3, "creation": 0.5, "nature": 0.5, "order": 0.5},
    "yah": {"authority": 0.9, "transcendence": 0.9, "care": 0.6, "justice": 0.9, "wisdom": 0.9, "power": 0.9, "fertility": 0.2, "war": 0.4, "death": 0.3, "creation": 0.9, "nature": 0.3, "order": 0.9},
    "baal": {"authority": 0.7, "transcendence": 0.6, "care": 0.4, "justice": 0.6, "wisdom": 0.5, "power": 0.8, "fertility": 0.6, "war": 0.6, "death": 0.3, "creation": 0.5, "nature": 0.7, "order": 0.6},
}

# Normalize all deity priors to unit vectors
for name, vec in DEITY_PRIORS.items():
    n = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    for k in vec:
        vec[k] /= n

VOWELS = set("aeiou")


def rnd_unit_vec(rng: random.Random) -> Dict[str, float]:
    v = {a: rng.random() for a in AXES}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {k: x / n for k, x in v.items()}


def dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(a[k] * b[k] for k in AXES)


def norm(a: Dict[str, float]) -> float:
    return math.sqrt(sum(a[k] * a[k] for k in AXES))


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def add_scaled(a: Dict[str, float], b: Dict[str, float], s: float) -> None:
    for k in AXES:
        a[k] += b[k] * s


def jitter_vec(base: Dict[str, float], rng: random.Random, noise: float = 0.1) -> Dict[str, float]:
    """Add Gaussian noise to a vector and renormalize"""
    v = {k: base[k] + rng.gauss(0, noise) for k in AXES}
    n = norm(v) or 1.0
    return {k: v[k] / n for k in v.keys()}


def scale(a: Dict[str, float], s: float) -> Dict[str, float]:
    return {k: a[k] * s for k in AXES}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------- Config ---------------------------- #

@dataclass
class Config:
    N: int = 64                            # agents
    steps_per_generation: int = 2000
    max_message_len: int = 3
    learning_rate: float = 0.08
    penalty_rate: float = 0.02
    prestige_alpha: float = 0.20           # amplification on attention
    ritual_period: int = 50                # steps between ritual frames
    ritual_bonus: float = 0.10             # improves success threshold
    base_success_thresh: float = 0.58      # cosine similarity threshold
    mutation_rate: float = 0.08            # per agent per step
    compounding_rate: float = 0.02         # chance to compound two forms
    lenition_rate: float = 0.25
    redup_rate: float = 0.10
    vowel_drop_rate: float = 0.20
    clitic_rate: float = 0.08
    exploration_eps: float = 0.10          # softmax exploration in production
    generation_mix_k: int = 3              # mentors per new agent
    seed: int = 7
    track_bigrams: bool = True
    topo_window: int = 200                 # contexts kept for topo metric
    # New parameters for deity priors and social dynamics
    use_deity_priors: bool = True          # use DEITY_PRIORS vs random vectors
    belief_influence: float = 0.15         # weight of agent belief in production
    coercion: float = 0.0                  # 0=polytheistic, 1=monotheistic pressure
    social_network: str = "small_world"    # "random", "small_world", "preferential"
    social_k: int = 4                      # small-world graph param
    social_p: float = 0.1                  # small-world graph param
    cluster_update_freq: int = 100         # steps between clustering updates
    cluster_threshold: float = 0.4         # distance to form new cluster
    # Accessibility Corollary parameters
    enable_sensory_restrictions: bool = False  # enable channel-restricted agents
    sensory_restriction_ratio: float = 0.2     # fraction of agents with restrictions
    channel_noise: float = 0.1                 # noise in channel projections


# ---------------------------- World / Context ---------------------------- #

@dataclass
class Context:
    task: str
    role: str
    place: str
    tod: str  # time of day
    vec: Dict[str, float]


class GridWorld:
    TASKS = ("forage", "warn", "trade", "mourn", "build", "raid")
    ROLES = ("leader", "hunter", "healer", "crafter")
    PLACES = ("camp", "river", "forest", "market", "grave")
    TODS = ("dawn", "noon", "dusk", "night")

    def __init__(self, cfg: Config, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.step_idx = 0
        self.ritual_frame = False

    def sample_context(self) -> Context:
        r = self.rng
        task, role, place, tod = (
            r.choice(self.TASKS), r.choice(self.ROLES), r.choice(self.PLACES), r.choice(self.TODS)
        )
        # encode to semantics (hand‑crafted biases)
        v = {k: 0.0 for k in AXES}
        if task == "forage":
            v["care"] += 0.4; v["power"] += 0.3; v["nature"] += 0.1
        elif task == "warn":
            v["authority"] += 0.4; v["wisdom"] += 0.2; v["justice"] += 0.2
        elif task == "trade":
            v["justice"] += 0.3; v["authority"] += 0.2; v["care"] += 0.2
        elif task == "mourn":
            v["death"] += 0.5; v["transcendence"] += 0.2; v["wisdom"] += 0.2
        elif task == "build":
            v["creation"] += 0.4; v["order"] += 0.2; v["power"] += 0.2
        elif task == "raid":
            v["war"] += 0.3; v["justice"] += 0.3; v["power"] += 0.2

        if role == "leader":
            v["authority"] += 0.3
        elif role == "healer":
            v["care"] += 0.3; v["wisdom"] += 0.2
        elif role == "hunter":
            v["power"] += 0.3; v["nature"] += 0.2
        elif role == "crafter":
            v["creation"] += 0.3

        if place == "grave":
            v["death"] += 0.3; v["transcendence"] += 0.2
        elif place == "market":
            v["justice"] += 0.3
        elif place == "river":
            v["nature"] += 0.2

        if tod in ("dusk", "night"):
            v["transcendence"] += 0.2
        else:
            v["order"] += 0.1

        # normalize
        n = norm(v) or 1.0
        v = {k: v[k] / n for k in AXES}
        return Context(task, role, place, tod, v)

    def tick(self):
        self.step_idx += 1
        self.ritual_frame = (self.step_idx % self.cfg.ritual_period == 0)


# ---------------------------- Agents ---------------------------- #

@dataclass
class Agent:
    id: int
    belief: Dict[str, float]
    w: float
    # Association: form -> semantic vector
    assoc: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Frequency of forms (useful for softmax prior)
    freq: Counter = field(default_factory=Counter)
    # Accessibility Corollary: sensory channel restrictions
    sensory_channels: List[str] = field(default_factory=lambda: list(SENSORY_CHANNELS.keys()))
    
    def project_context(self, context_vec: Dict[str, float], noise: float = 0.0) -> Dict[str, float]:
        """Project context through available sensory channels (φ_s mapping)"""
        if len(self.sensory_channels) == len(SENSORY_CHANNELS):
            # No restrictions - return full context
            return context_vec.copy()
        
        # Restricted channels - project only through available modalities
        projected = {k: 0.0 for k in AXES}
        channel_count = 0
        
        for channel in self.sensory_channels:
            if channel in SENSORY_CHANNELS:
                channel_axes = SENSORY_CHANNELS[channel]
                for axis in channel_axes:
                    if axis in context_vec:
                        projected[axis] += context_vec[axis]
                        channel_count += 1
        
        # Normalize by number of available channels (b_i = 1/|S_i| * sum)
        if channel_count > 0:
            for axis in projected:
                projected[axis] /= len(self.sensory_channels)
        
        # Add channel noise if specified
        if noise > 0.0:
            import random
            for axis in projected:
                projected[axis] += random.gauss(0, noise)
        
        # Renormalize
        n = norm(projected) or 1.0
        return {k: projected[k] / n for k in AXES}

    def known_forms(self) -> Iterable[str]:
        return self.assoc.keys()


# ---------------------------- Morphology (mutations) ---------------------------- #

class Morphology:
    def __init__(self, cfg: Config, rng: random.Random):
        self.cfg = cfg
        self.rng = rng

    def mutate(self, forms: List[str]) -> List[str]:
        r = self.rng
        out = list(forms)
        if r.random() < self.cfg.compounding_rate and len(forms) >= 2:
            a, b = r.sample(forms, 2)
            out.append(f"{a}-{b}")
        return [self._mutate_form(f) for f in out]

    def _mutate_form(self, f: str) -> str:
        r = self.rng
        s = f
        if r.random() < self.cfg.vowel_drop_rate:
            s = ''.join(ch for ch in s if ch not in VOWELS) or s
        if r.random() < self.cfg.lenition_rate:
            s = (s.replace('t', 'd').replace('k', 'g').replace('p', 'b')
                   .replace('s', 'h').replace('f', 'v'))
        if r.random() < self.cfg.redup_rate:
            s = s + s[-1]
        if r.random() < self.cfg.clitic_rate:
            s = s + "="
        return s


# ---------------------------- Kernel ---------------------------- #

@dataclass
class Metrics:
    zipf_slope: float = 0.0
    heaps_k: float = 0.0
    cond_entropy: float = 0.0
    topo_similarity: float = 0.0
    churn: float = 0.0   # rough: switching rate of preferred forms


@dataclass
class TickResult:
    t: int
    success: bool
    ritual: bool


class SwarmKernel:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.world = GridWorld(cfg, self.rng)
        self.morph = Morphology(cfg, self.rng)
        self.agents: List[Agent] = []
        self._init_agents()
        self.t = 0
        self.gen = 0

        # telemetry
        self.tokens: List[str] = []
        self.types: Counter = Counter()
        self.bigrams: Counter = Counter()
        self.last_token: Optional[str] = None
        self.ctx_window: deque[Tuple[Dict[str, float], Dict[str, float]]] = deque(maxlen=cfg.topo_window)
        self.pref_form: Dict[int, Optional[str]] = {a.id: None for a in self.agents}
        self.metrics = Metrics()
        self.log: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.social_graph: Dict[int, List[int]] = {}
        self.clusters: List[List[int]] = []
        self.centroids: List[Dict[str, float]] = []

    # ---------- initialization ---------- #
    def _init_agents(self):
        for i in range(self.cfg.N):
            # Initialize belief from deity mixture if using priors
            if self.cfg.use_deity_priors:
                # Pick 1-2 deities for this agent's belief foundation
                chosen = self.rng.sample(list(DEITY_PRIORS.keys()), k=self.rng.randint(1, 2))
                belief = {k: 0.0 for k in AXES}
                for deity in chosen:
                    add_scaled(belief, DEITY_PRIORS[deity], 1.0 / len(chosen))
                belief = jitter_vec(belief, self.rng, 0.15)
            else:
                belief = rnd_unit_vec(self.rng)
            
            a = Agent(id=i, belief=belief, w=1.0)
            
            # Accessibility Corollary: assign sensory restrictions
            if self.cfg.enable_sensory_restrictions and self.rng.random() < self.cfg.sensory_restriction_ratio:
                # Create sensory-restricted agents (deafblind, deaf, blind, etc.)
                restriction_type = self.rng.choice(["deafblind", "deaf", "blind", "anosmia"])
                if restriction_type == "deafblind":
                    a.sensory_channels = ["touch", "proprioception", "smell", "taste"]
                elif restriction_type == "deaf":
                    a.sensory_channels = ["sight", "touch", "proprioception", "smell", "taste"]
                elif restriction_type == "blind":
                    a.sensory_channels = ["sound", "touch", "proprioception", "smell", "taste"]
                elif restriction_type == "anosmia":
                    a.sensory_channels = ["sight", "sound", "touch", "proprioception", "taste"]
            
            # seed each with the base theonyms + deity priors if enabled
            for name in set(THEONYMS):
                if self.cfg.use_deity_priors and name in DEITY_PRIORS:
                    a.assoc[name] = jitter_vec(DEITY_PRIORS[name], self.rng, 0.1)
                else:
                    a.assoc[name] = rnd_unit_vec(self.rng)
            self.agents.append(a)
        self._build_social_network()

    # ---------- social network ---------- #
    def _build_social_network(self):
        # builds a Watts-Strogatz small-world graph
        self.social_graph = {i: [] for i in range(self.cfg.N)}
        k = self.cfg.social_k // 2
        for i in range(self.cfg.N):
            for j in range(1, k + 1):
                neighbor = (i + j) % self.cfg.N
                self.social_graph[i].append(neighbor)
                self.social_graph[neighbor].append(i)
        # rewire
        for i in range(self.cfg.N):
            for j in range(len(self.social_graph[i])):
                if self.rng.random() < self.cfg.social_p:
                    neighbor = self.social_graph[i][j]
                    self.social_graph[i].remove(neighbor)
                    self.social_graph[neighbor].remove(i)
                    new_neighbor = self.rng.choice([n for n in range(self.cfg.N) if n != i and n not in self.social_graph[i]])
                    self.social_graph[i].append(new_neighbor)
                    self.social_graph[new_neighbor].append(i)

    def _select_hearers(self, speaker: Agent) -> List[Agent]:
        neighbors = self.social_graph.get(speaker.id, [])
        if not neighbors:
            return self.rng.sample([a for a in self.agents if a != speaker], k=min(2, self.cfg.N - 1))

        if self.cfg.coercion > 0:
            # weight neighbors by belief similarity to speaker
            weights = []
            for nid in neighbors:
                hearer = self.agents[nid]
                sim = cosine(speaker.belief, hearer.belief)
                # coercion acts as sharpness parameter
                weight = math.exp(sim * (1 + 9 * self.cfg.coercion))
                weights.append(weight)
            
            chosen_ids = self.rng.choices(neighbors, weights=weights, k=min(2, len(neighbors)))
            return [self.agents[i] for i in chosen_ids]
        else:
            chosen_ids = self.rng.sample(neighbors, k=min(2, len(neighbors)))
            return [self.agents[i] for i in chosen_ids]

    # ---------- production/interpretation ---------- #
    def _softmax_choice(self, items: List[Tuple[str, float]]) -> str:
        # exploration‑enhanced softmax
        eps = self.cfg.exploration_eps
        if self.rng.random() < eps:
            return self.rng.choice([k for k, _ in items])
        mx = max(s for _, s in items) if items else 1.0
        exps = [(k, math.exp(s - mx)) for k, s in items]
        z = sum(v for _, v in exps) or 1.0
        r = self.rng.random() * z
        acc = 0.0
        for k, v in exps:
            acc += v
            if acc >= r:
                return k
        return items[-1][0]

    def produce(self, agent: Agent, ctx: Context) -> List[str]:
        # score forms by association dot product with context + frequency prior + belief influence
        scored: List[Tuple[str, float]] = []
        total = sum(agent.freq.values()) + 1
        for form, vec in agent.assoc.items():
            ctx_score = dot(vec, ctx.vec)
            belief_score = dot(vec, agent.belief) * self.cfg.belief_influence
            freq_score = 0.1 * math.log((agent.freq[form] + 1) / total)
            s = ctx_score + belief_score + freq_score
            scored.append((form, s))
        msg = []
        for _ in range(self.cfg.max_message_len):
            choice = self._softmax_choice(scored)
            msg.append(choice)
        return msg

    def interpret(self, hearer: Agent, msg: List[str]) -> Dict[str, float]:
        # bag‑of‑forms embedding → predicted context vector
        v = {k: 0.0 for k in AXES}
        for tok in msg:
            if tok in hearer.assoc:
                add_scaled(v, hearer.assoc[tok], 1.0)
        n = norm(v)
        if n == 0:
            return rnd_unit_vec(self.rng)
        return {k: v[k] / n for k in AXES}

    def interact(self, speaker: Agent, hearers: List[Agent], ctx: Context, msg: List[str]) -> bool:
        # Apply sensory channel projections for each hearer
        preds = []
        for h in hearers:
            projected_ctx = h.project_context(ctx.vec, self.cfg.channel_noise)
            pred = self.interpret(h, msg)
            preds.append(pred)
        
        sim = statistics.mean(cosine(ctx.vec, p) for p in preds)
        thresh = self.cfg.base_success_thresh + (self.cfg.ritual_bonus if self.world.ritual_frame else 0.0)
        return sim >= thresh

    # ---------- learning & prestige ---------- #
    def learn_from(self, agent: Agent, msg: List[str], ctx: Context, success: bool):
        lr = self.cfg.learning_rate if success else -self.cfg.penalty_rate
        # Apply agent's sensory projection to context for learning
        projected_ctx = agent.project_context(ctx.vec, self.cfg.channel_noise)
        # Blend projected context with agent belief for learning target
        blend_weight = 0.2  # how much belief influences learning
        ctx_blend = {k: (1-blend_weight)*projected_ctx[k] + blend_weight*agent.belief[k] for k in AXES}
        
        for tok in msg:
            if tok not in agent.assoc:
                if self.cfg.use_deity_priors and tok in DEITY_PRIORS:
                    agent.assoc[tok] = jitter_vec(DEITY_PRIORS[tok], self.rng, 0.1)
                else:
                    agent.assoc[tok] = rnd_unit_vec(self.rng)
            add_scaled(agent.assoc[tok], ctx_blend, lr)
        for tok in msg:
            agent.freq[tok] += 1

    def update_prestige(self, agents: Iterable[Agent], success: bool):
        for a in agents:
            delta = (self.cfg.prestige_alpha if success else -self.cfg.prestige_alpha * 0.3)
            a.w = clamp(a.w * (1.0 + delta), 0.1, 10.0)

    # ---------- clustering ---------- #
    def _update_clusters(self):
        # online clustering: assign agents to nearest centroid or create new one
        self.centroids = []
        self.clusters = []

        for agent in self.agents:
            if not self.centroids:
                self.centroids.append(agent.belief.copy())
                self.clusters.append([agent.id])
                continue

            distances = [1 - cosine(agent.belief, c) for c in self.centroids]
            min_dist = min(distances)
            best_idx = distances.index(min_dist)

            if min_dist < self.cfg.cluster_threshold:
                self.clusters[best_idx].append(agent.id)
            else:
                self.centroids.append(agent.belief.copy())
                self.clusters.append([agent.id])

        # recalculate centroids
        new_centroids = []
        for i, cluster in enumerate(self.clusters):
            if not cluster:
                continue
            centroid = {k: 0.0 for k in AXES}
            for agent_id in cluster:
                add_scaled(centroid, self.agents[agent_id].belief, 1.0)
            new_centroids.append(scale(centroid, 1.0 / len(cluster)))
        self.centroids = new_centroids

    # ---------- mutation & drift ---------- #
    def mutate_agent(self, agent: Agent):
        if self.rng.random() >= self.cfg.mutation_rate:
            return
        forms = list(agent.known_forms())
        if not forms:
            return
        new_forms = self.morph.mutate(self.rng.sample(forms, min(2, len(forms))))
        for nf in new_forms:
            if nf not in agent.assoc:
                agent.assoc[nf] = rnd_unit_vec(self.rng)

    # ---------- generation / transmission ---------- #
    def generation_boundary(self) -> bool:
        return (self.t > 0 and self.t % self.cfg.steps_per_generation == 0)

    def transmit(self):
        # pick mentors weighted by prestige
        weights = [a.w for a in self.agents]
        total = sum(weights) or 1.0
        probs = [w / total for w in weights]
        mentors = self.rng.choices(self.agents, weights=probs, k=self.cfg.generation_mix_k)

        new_agents: List[Agent] = []
        for i in range(self.cfg.N):
            belief = rnd_unit_vec(self.rng)
            na = Agent(id=i, belief=belief, w=1.0)
            # merge a subset of mentors' associations/frequencies
            for m in mentors:
                items = list(m.assoc.items())
                if items:
                    sample_k = min(10, len(items))
                    for form, vec in self.rng.sample(items, k=sample_k):
                        if form not in na.assoc:
                            na.assoc[form] = dict(vec)
                            na.freq[form] += m.freq[form]
            # ensure base theonyms exist
            for name in set(THEONYMS):
                na.assoc.setdefault(name, rnd_unit_vec(self.rng))
            new_agents.append(na)
        self.agents = new_agents
        self.pref_form = {a.id: None for a in self.agents}
        self.gen += 1

    # ---------- metrics ---------- #
    def _update_token_stats(self, msg: List[str]):
        for tok in msg:
            self.tokens.append(tok)
            self.types[tok] += 1
            if self.cfg.track_bigrams and self.last_token is not None:
                self.bigrams[(self.last_token, tok)] += 1
            self.last_token = tok

    def _zipf_slope(self) -> float:
        if not self.types:
            return 0.0
        freqs = sorted(self.types.values(), reverse=True)
        ranks = list(range(1, len(freqs) + 1))
        x = [math.log(r) for r in ranks]
        y = [math.log(f) for f in freqs]
        n = len(x)
        sx = sum(x); sy = sum(y)
        sxx = sum(v*v for v in x); sxy = sum(x[i]*y[i] for i in range(n))
        denom = n*sxx - sx*sx
        if denom == 0:
            return 0.0
        slope = (n*sxy - sx*sy) / denom
        return slope  # expect ~ -1 for natural language

    def _heaps_k(self) -> float:
        # types ≈ K * tokens^b ; we return K estimator for b≈0.5 as proxy
        T = len(self.tokens)
        V = len(self.types)
        if T == 0:
            return 0.0
        b = 0.5
        return V / (T ** b)

    def _cond_entropy(self) -> float:
        if not self.bigrams:
            return 0.0
        # H(next | prev) = -sum p(prev) sum p(next|prev) log2 p(next|prev)
        prev_counts = Counter()
        for (p, n), c in self.bigrams.items():
            prev_counts[p] += c
        H = 0.0
        total_pairs = sum(self.bigrams.values())
        for p, pc in prev_counts.items():
            inner = 0.0
            for (pp, nn), c in self.bigrams.items():
                if pp == p:
                    q = c / pc
                    inner += -q * math.log(q + 1e-12, 2)
            H += (pc / total_pairs) * inner
        return H

    def _topo_similarity(self) -> float:
        # correlation between context distances and message embedding distances
        if len(self.ctx_window) < 8:
            return 0.0
        pairs = list(itertools.combinations(range(len(self.ctx_window)), 2))
        ctx_d: List[float] = []
        msg_d: List[float] = []
        for i, j in self.rng.sample(pairs, k=min(len(pairs), 200)):
            cv_i, mv_i = self.ctx_window[i]
            cv_j, mv_j = self.ctx_window[j]
            ctx_d.append(1 - cosine(cv_i, cv_j))
            msg_d.append(1 - cosine(mv_i, mv_j))
        # Pearson r
        n = len(ctx_d)
        if n < 2:
            return 0.0
        mx, my = statistics.mean(ctx_d), statistics.mean(msg_d)
        sx = math.sqrt(sum((x - mx)**2 for x in ctx_d)) or 1.0
        sy = math.sqrt(sum((y - my)**2 for y in msg_d)) or 1.0
        cov = sum((ctx_d[i] - mx)*(msg_d[i] - my) for i in range(n))
        return cov / (sx * sy)

    def _update_metrics(self):
        self.metrics.zipf_slope = self._zipf_slope()
        self.metrics.heaps_k = self._heaps_k()
        self.metrics.cond_entropy = self._cond_entropy()
        self.metrics.topo_similarity = self._topo_similarity()
        # churn: fraction of agents whose preferred token changed since last check
        changed = 0
        for a in self.agents:
            # preferred form = most frequent token for this agent
            if a.freq:
                new_pref = max(a.freq.items(), key=lambda kv: kv[1])[0]
            else:
                new_pref = None
            if new_pref != self.pref_form.get(a.id):
                changed += 1
            self.pref_form[a.id] = new_pref
        self.metrics.churn = changed / max(1, len(self.agents))

        # log
        self.log["zipf"].append((self.t, self.metrics.zipf_slope))
        self.log["heaps_k"].append((self.t, self.metrics.heaps_k))
        self.log["entropy"].append((self.t, self.metrics.cond_entropy))
        self.log["topo"].append((self.t, self.metrics.topo_similarity))
        self.log["churn"].append((self.t, self.metrics.churn))

    # ---------- main loop ---------- #
    def step(self) -> TickResult:
        self.t += 1
        self.world.tick()
        ctx = self.world.sample_context()
        speaker = self.rng.choices(self.agents, weights=[a.w for a in self.agents], k=1)[0]
        hearers = self._select_hearers(speaker)
        msg = self.produce(speaker, ctx)

        # update global token stats
        self._update_token_stats(msg)

        success = self.interact(speaker, hearers, ctx, msg)
        self.update_prestige([speaker] + hearers, success)
        # learning happens for both speaker and hearers
        self.learn_from(speaker, msg, ctx, success)
        for h in hearers:
            self.learn_from(h, msg, ctx, success)
        # record for topo sim: store (context vec, message embedding vec)
        m_emb = self.interpret(speaker, msg)
        self.ctx_window.append((ctx.vec, m_emb))

        # mutations
        for a in [speaker] + hearers:
            self.mutate_agent(a)

        # metrics
        if self.t % 25 == 0:
            self._update_metrics()
        
        # clustering
        if self.t % self.cfg.cluster_update_freq == 0:
            self._update_clusters()

        # generation
        if self.generation_boundary():
            self.transmit()

        return TickResult(t=self.t, success=success, ritual=self.world.ritual_frame)

    def run(self, steps: int, callbacks: Optional["Callbacks"] = None):
        cb = callbacks or Callbacks()
        for _ in range(steps):
            res = self.step()
            cb.on_step(res.t, self)
            if self.generation_boundary():
                cb.on_generation(self.gen, self)

    # ---------- API ---------- #
    def snapshot(self) -> Dict:
        return {
            "t": self.t,
            "gen": self.gen,
            "metrics": asdict(self.metrics),
            "top_forms": Counter(self.tokens).most_common(12),
            "agents": [
                {
                    "id": a.id,
                    "w": round(a.w, 3),
                    "vocab": len(a.assoc),
                    "top": a.freq.most_common(3),
                }
                for a in self.agents[:10]
            ],
        }

    def seed_rng(self, seed: int):
        self.rng.seed(seed)


# ---------------------------- Callbacks ---------------------------- #

class Callbacks:
    def on_step(self, t: int, kernel: SwarmKernel):
        pass

    def on_generation(self, gen: int, kernel: SwarmKernel):
        pass


# ---------------------------- Demo ---------------------------- #

if __name__ == "__main__":
    cfg = Config()
    kern = SwarmKernel(cfg)

    class PrintCB(Callbacks):
        def on_step(self, t: int, k: SwarmKernel):
            if t % 500 == 0:
                snap = k.snapshot()
                m = snap["metrics"]
                print(
                    f"t={snap['t']:5d} gen={snap['gen']} | "
                    f"zipf={m['zipf_slope']:+.2f} heapsK={m['heaps_k']:.3f} "
                    f"H={m['cond_entropy']:.2f} topo={m['topo_similarity']:+.2f} "
                    f"churn={m['churn']:.2f} | top={snap['top_forms'][:5]}"
                )
        def on_generation(self, gen: int, k: SwarmKernel):
            print(f"-- generation {gen} --")

    kern.run(steps=5000, callbacks=PrintCB())
    print(json.dumps(kern.snapshot(), indent=2))
