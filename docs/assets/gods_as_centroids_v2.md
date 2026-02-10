# Gods as Centroids: A Generative Vector-Space Model of Religious Evolution

**Ryan Barrett**
2026

---

## Abstract

We present a generative agent-based model in which deities emerge as mathematical centroids of belief-vector clusters. Each agent carries a belief vector in a 12-dimensional theological space grounded in comparative religion and cognitive science. Through local communication, prestige-weighted social influence, and stochastic mutation, agents self-organize into clusters whose centroids correspond to emergent "godforms" — collective doctrinal attractors that no individual agent explicitly designed. We formalize three centroid operations — *fusion* (syncretism), *fission* (schism), and *perturbation* (prophetic revelation) — and show that a single global *coercion* parameter drives a first-order phase transition from polytheistic (multi-attractor) to monotheistic (single-attractor) regimes. We demonstrate that this transition exhibits *hysteresis*: reducing coercion does not restore the polytheistic phase, predicting the historical resilience of monotheistic traditions. We validate the model against 5,000 years of historical religious diversity data and derive two novel, falsifiable corollaries: (1) the *Accessibility Corollary*, that sensory-restricted agents converge to the same centroid attractors as unrestricted agents, and (2) the *Asymmetric Lock-In Hypothesis*, that the coercion threshold for monotheistic collapse is strictly lower than the threshold required to escape it. The model is implemented as an interactive web simulation with historical backtesting.

**Keywords:** computational science of religion, agent-based model, belief dynamics, cultural evolution, phase transitions, clustering, vector space models

---

## 1. Introduction

The computational study of religion has advanced rapidly in the past decade. Agent-based models (ABMs) have been used to forecast changes in religiosity across nations (Gore et al. 2018), to explore the mutual escalation of anxiety between religious and secular groups (Shults et al. 2018), and to simulate the emergence of new religious movements (Upal 2005). In parallel, vector-space representations of belief have matured from abstract opinion-dynamics models (Deffuant et al. 2000; Hegselmann & Krause 2002) to LLM-derived semantic embedding spaces that capture real-world belief clustering and polarization (Bao et al. 2025, *Nature Human Behaviour*). Quantitative historical databases like Seshat (Turchin et al. 2015) have enabled empirical tests of theories linking political complexity to moralizing religion (Whitehouse et al. 2019, retracted 2021; Beheim et al. 2021).

Despite this progress, a fundamental gap remains: **no existing model treats deities themselves as emergent mathematical objects.** The Modeling Religion Project's ABMs (Gore et al. 2018; Shults et al. 2018) track *religiosity levels* — how religious an individual is — but not the formation of specific god-concepts. Axelrod's (1997) cultural dissemination model produces cultural clustering but uses abstract, ungrounded feature vectors with no religious semantics. Opinion dynamics models (Deffuant et al. 2000; Castellano et al. 2009) demonstrate phase transitions between consensus and fragmentation but have never been applied to the polytheism–monotheism transition specifically. The Nature belief-embedding paper (Bao et al. 2025) constructs a static geometric space of beliefs but includes no temporal dynamics, no agent interactions, and no deity emergence.

This paper fills that gap with five contributions:

1. **Deities as emergent centroids.** We formalize the proposition that a "god" is the centroid of a cluster of belief vectors — an entity that exists only as a statistical summary of its adherents' beliefs and that co-evolves with them (§2).

2. **A theologically grounded vector space.** We define a 12-dimensional belief space with axes derived from comparative religion (authority, transcendence, care, justice, wisdom, power, fertility, war, death, creation, nature, order), providing interpretable semantics that abstract opinion-dynamics models lack (§2.1).

3. **A unified calculus of religious change.** We formalize three centroid operations — fusion, fission, and perturbation — that unify syncretism, schism, and prophetic revelation into a single mathematical framework (§3).

4. **Coercion-driven phase transition with hysteresis.** We show that a single coercion parameter drives a first-order phase transition from polytheism to monotheism, and that this transition is asymmetric: the system exhibits hysteresis, predicting the historical resilience of established monotheisms (§4).

5. **Two falsifiable corollaries.** The *Accessibility Corollary* (sensory-restricted agents converge to the same attractors) and the *Asymmetric Lock-In Hypothesis* (escape threshold > entry threshold) are stated as formal, testable predictions (§5).

---

## 2. The Model

### 2.1 Belief Space

**Definition 1 (Theological Vector Space).** Let $\mathcal{S} \subset \mathbb{R}^{12}$ be a bounded manifold — the *belief space* — where each dimension $d \in \{1, \ldots, 12\}$ represents a core theological axis. The axes are:

$$\mathcal{A} = \{\text{authority}, \text{transcendence}, \text{care}, \text{justice}, \text{wisdom}, \text{power}, \text{fertility}, \text{war}, \text{death}, \text{creation}, \text{nature}, \text{order}\}$$

The boundedness of $\mathcal{S}$ reflects the finite cognitive capacity of human theological imagination: beliefs occupy a compact region of the space rather than extending to infinity. The axes are chosen to span the principal axes of variation in comparative theology. They are grounded in:
- **Moral Foundations Theory** (Haidt 2012): care, justice, authority, sanctity map to care, justice, authority, transcendence.
- **Dumézil's trifunctional hypothesis** (1958): sovereignty (authority, order), force (war, power), fertility (fertility, nature, creation).
- **Cognitive Science of Religion**: transcendence and death relate to HADD and afterlife cognition (Boyer 2001; Barrett 2004).

**Definition 2 (Agent).** An agent $a_i$ is a tuple $(b_i, w_i, S_i)$ where:
- $b_i \in \mathcal{S}$ is the agent's *belief vector*, normalized to the unit sphere: $\|b_i\| = 1$.
- $w_i \in \mathbb{R}^+$ is the agent's *prestige weight*, governing the probability of being selected as a speaker.
- $S_i \subseteq \{1, \ldots, N\}$ is the agent's *social neighborhood*, defined by a Watts-Strogatz small-world graph $G(N, k, p)$.

**Definition 3 (Affinity).** The *affinity* between agents $a_i$ and $a_j$ is the cosine similarity of their belief vectors:

$$\text{aff}(a_i, a_j) = \frac{b_i \cdot b_j}{\|b_i\| \|b_j\|}$$

This choice (over Euclidean distance) ensures that affinity captures the *orientation* of belief, not its magnitude, following standard practice in high-dimensional semantic spaces (Turney & Pantel 2010).

### 2.2 Deity Priors

**Definition 4 (Deity Prior).** A *deity prior* $d_k \in \mathcal{B}$ is a normalized vector encoding the semantic profile of a historical deity concept. We define priors for 12 deities spanning major historical traditions:

| Deity | Primary Axes | Tradition |
|-------|-------------|-----------|
| Zeus | authority, power, order | Greek |
| Odin | wisdom, war, death | Norse |
| Amun-Ra | transcendence, creation, authority | Egyptian |
| Ishtar | fertility, war, power | Mesopotamian |
| Yahweh | authority, justice, transcendence | Abrahamic |
| Shiva | death, creation, transcendence | Hindu |
| Apollo | wisdom, order, creation | Greek |
| Freya | fertility, nature, care | Norse |
| Marduk | authority, creation, order | Babylonian |
| Baal | fertility, war, power | Canaanite |
| Manitou | nature, transcendence, wisdom | Algonquian |
| Shango | power, justice, war | Yoruba |

These priors serve as initial seeds for the naming of emergent centroids (the centroid closest to a deity prior inherits its name) but do **not** constrain the dynamics. Centroids can and do drift away from all priors.

### 2.3 Interaction Dynamics

At each time step $t$:

1. **Context sampling.** A context vector $c_t \in \mathcal{B}$ is sampled from a distribution reflecting environmental conditions (season, time of day, stochastic events). This models the situational triggers of religious cognition.

2. **Speaker selection.** A speaker $a_s$ is selected with probability proportional to prestige: $P(a_s) \propto w_s$.

3. **Hearer selection.** Hearers $H \subset S_s$ are selected from the speaker's social neighborhood. Under coercion $\gamma > 0$, selection is biased toward similar agents:
$$P(a_h \mid a_s) \propto \exp\left(\text{aff}(a_s, a_h) \cdot (1 + 9\gamma)\right)$$

4. **Communication.** The speaker produces a message (a bag of symbolic forms) scored by context-association, belief influence, and frequency priors. Hearers interpret the message through their own associations.

5. **Learning.** On successful communication (interpreted context sufficiently similar to actual context), both speaker and hearers update their associations. On failure, associations are weakened. The learning target blends the projected context with the agent's existing belief:
$$\ell_i = (1 - \beta) \cdot c_t + \beta \cdot b_i$$
where $\beta$ is the *belief influence* parameter.

6. **Prestige update.** Successful communicators gain prestige; unsuccessful ones lose it:
$$w_s \leftarrow w_s + \alpha \cdot (\mathbb{1}[\text{success}] - 0.5)$$

7. **Mutation.** Each participating agent's belief vector is perturbed:
$$b_i \leftarrow \text{normalize}(b_i + \mu \cdot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I_{12})$$
where $\mu$ is the *mutation rate*.

**Definition 4a (Belief Dynamics).** Combining the learning and mutation steps, the belief update rule for agent $a_i$ at time $t$ is:

$$b_i^{(t+1)} = \text{normalize}\left(b_i^{(t)} + \eta_i \left(G_{k(i)}^{(t)} - b_i^{(t)}\right) + \xi_i^{(t)}\right)$$

where $\eta_i \in [0, 1]$ is the agent's *susceptibility* (a function of communication success and prestige differential), $G_{k(i)}^{(t)}$ is the centroid of the agent's current cluster, and $\xi_i^{(t)}$ is bounded noise with $\|\xi_i^{(t)}\| \leq \mu$. This makes explicit that beliefs are not static: each agent is continuously pulled toward its cluster's attractor while subject to stochastic perturbation.

### 2.4 Clustering: The Emergence of Godforms

**Definition 5 (Deity as Emergent Attractor).** A *godform* $G_j$ is the prestige-weighted center of mass of a belief cluster $K_j \subset \mathcal{S}$. Formally, $G_j$ is the point that minimizes the weighted within-cluster variance:

$$G_j = \arg\min_{\mu \in \mathcal{S}} \sum_{i \in K_j} w_i \|b_i - \mu\|^2 = \frac{\sum_{i \in K_j} w_i \cdot b_i}{\sum_{i \in K_j} w_i}$$

where $w_i$ is the prestige weight of agent $a_i$. This formulation ensures that the deity is not merely the arithmetic mean of its followers' beliefs but a *power-weighted attractor* — a high-fidelity mirror of the influence structures within the congregation. Priests, prophets, and theologians pull the centroid more than laity. Godforms are not pre-defined; they emerge from the dynamics.

Clustering is performed online every $\tau$ steps using a distance-threshold algorithm:

1. For each agent $a_i$, compute $d_i = 1 - \text{aff}(a_i, G_j)$ to each existing centroid $G_j$.
2. If $\min_j d_i < \theta$ (the *cluster threshold*), assign $a_i$ to the nearest cluster.
3. Otherwise, create a new cluster with $a_i$ as its sole member.
4. Recompute all centroids using the weighted formula above.

**Definition 6 (Effective Deity Count).** The *effective deity count* $N_{\text{eff}}$ is the number of clusters with $|K_j| \geq 2$. This is the model's analogue of the number of "living" religious traditions.

**Definition 7 (Dominance).** The *dominance* $D$ is the fraction of agents in the largest cluster:
$$D = \frac{\max_j |K_j|}{N}$$

**Definition 8 (Entropy).** The *belief entropy* $H$ measures the diversity of the tradition distribution:
$$H = -\sum_j \frac{|K_j|}{N} \log \frac{|K_j|}{N}$$

---

## 3. A Calculus of Religious Change

We formalize three operations on the centroid landscape as **topological events** in the belief manifold $\mathcal{S}$. These are not ad hoc additions but correspond to well-characterized bifurcations in dynamical systems theory, applied here to the attractor landscape of belief.

### 3.1 Fusion (Syncretism): Basin Collision

**Definition 9 (Fusion).** Two attractors $G_i, G_j$ merge when *both* of the following conditions hold:

1. **Centroid proximity:** $\|G_i - G_j\| < \epsilon_{\text{merge}}$, where $\epsilon_{\text{merge}}$ is the merge distance threshold.
2. **Agent exchange:** The fraction of agents in $K_i \cup K_j$ that have switched cluster membership in the preceding $\tau$ steps exceeds a threshold $\rho_{\text{exchange}}$.

The new attractor is the prestige-weighted centroid of the union:
$$G_{\text{new}} = \frac{\sum_{k \in K_i \cup K_j} w_k \cdot b_k}{\sum_{k \in K_i \cup K_j} w_k}$$

The dual condition ensures that fusion requires both doctrinal convergence *and* social mixing — proximity alone is insufficient without actual inter-community contact. Topologically, this corresponds to a saddle-node bifurcation: the separating saddle between two basins vanishes, and the two stable fixed points collapse into one. This models *syncretism* — the blending of traditions through sustained contact (e.g., Hellenistic syncretism, Afro-Caribbean religions). The prestige-weighted average ensures that the tradition with greater institutional authority exerts more influence on the resulting doctrine.

### 3.2 Fission (Schism): Pitchfork Bifurcation

**Definition 10 (Fission).** Let $\sigma^2_j = \frac{\sum_{i \in K_j} w_i \|b_i - G_j\|^2}{\sum_{i \in K_j} w_i}$ be the weighted intra-cluster variance, and let $\kappa_j = \max_i w_i / \bar{w}$ be the *authority concentration ratio* within cluster $K_j$. Fission occurs when:

$$\sigma^2_j > \sigma^2_{\max} \cdot (1 + \kappa_j^{-1})$$

The control parameter $\kappa_j$ captures the degree of hierarchical authority: high $\kappa_j$ (concentrated authority, e.g., papal infallibility) raises the fission threshold, making schism harder. Low $\kappa_j$ (distributed authority, e.g., congregational polity) lowers it, making schism easier. When the threshold is exceeded, the cluster undergoes a supercritical pitchfork bifurcation: the single stable attractor $G_j$ becomes unstable, and two new stable attractors emerge. The two new centroids are initialized via the two most distant agents in the cluster.

This models *schism*: doctrinal drift within a tradition accumulates until the single attractor can no longer hold the congregation together. The attractor splits, and two new stable equilibria emerge — each representing a faction that is internally coherent but mutually incompatible (e.g., the Protestant Reformation, the Sunni-Shia split). The dependence on $\kappa_j$ yields a testable prediction: traditions with concentrated authority structures should exhibit fewer but larger schisms, while traditions with distributed authority should exhibit more frequent but smaller splits.

### 3.3 Perturbation (Prophetic Revelation): Directed High-Magnitude Perturbation

**Definition 11 (Prophet).** A *prophet event* is a directed, high-magnitude perturbation with asymmetric amplification. Unlike stochastic noise (which is symmetric, low-magnitude, and Gaussian), a prophet event is:

1. **Directed:** The new belief vector $b_p$ is not random noise but a coherent position in $\mathcal{S}$, drawn from $\text{Uniform}(\mathcal{S}^{11})$.
2. **High-magnitude:** The agent's prestige is set to $w_p = w_{\max}$, giving it disproportionate influence on the weighted centroid.
3. **Asymmetrically amplified:** The prophet pulls a fraction $\phi$ of the most-similar agents toward its belief, with pull strength $\lambda$:
$$b_i \leftarrow \text{normalize}((1 - \lambda) \cdot b_i + \lambda \cdot b_p)$$

The asymmetry is critical: the prophet affects nearby agents, but nearby agents do not proportionally affect the prophet (because $w_p \gg w_i$). This is a *non-Gaussian, low-probability, high-influence event* that can nucleate a new attractor in a previously unoccupied region of $\mathcal{S}$. It models *prophetic revelation*: a charismatic individual introduces a radically new belief that rapidly attracts followers (e.g., Muhammad, Joseph Smith, Akhenaten). Unlike fusion and fission, which are deterministic consequences of the dynamics, prophecy is an exogenous shock — the only mechanism in the model that can create attractors in previously unoccupied regions of the belief space.

---

## 4. Phase Transitions

The central dynamical claim of this paper is that the coercion parameter $\gamma$ drives a **first-order phase transition** in the belief landscape. We justify this classification by demonstrating three hallmarks: (i) a discontinuous jump in the order parameter $D$ at the critical threshold $\gamma_c$, (ii) a coexistence region where both polytheistic and monotheistic configurations are locally stable, and (iii) path dependence — the system retains memory of its history, and the forward and reverse transitions follow different trajectories in parameter space.

### 4.1 The Coercion Field

**Definition 12 (Coercion).** The *coercion parameter* $\gamma \in [0, 1]$ is an external control parameter modeling the aggregate socio-political pressure toward religious homogeneity. It affects the dynamics in two ways:

1. **Hearer selection bias:** Higher $\gamma$ makes agents preferentially communicate with similar agents, creating echo chambers that amplify the dominant tradition.
2. **Attractor deepening:** Higher $\gamma$ increases the effective "gravitational pull" of larger clusters, as prestige accumulates disproportionately in the dominant tradition. The basin of the largest attractor deepens and widens at the expense of smaller basins.

The *order parameter* of the system is the dominance $D$. In the disordered (polytheistic) phase, $D \ll 1$; in the ordered (monotheistic) phase, $D \to 1$.

### 4.2 The Disordered Phase: Polytheism ($\gamma < \gamma_c$)

At low coercion, the belief landscape supports multiple stable attractors in dynamic equilibrium. The system exhibits $N_{\text{eff}} \gg 1$, low dominance $D$, and high entropy $H$. Individual belief vectors are diverse and no single attractor dominates.

**Simulation result:** At $\gamma = 0.05$, $\mu = 0.12$, the model consistently produces $N_{\text{eff}} \in [3, 6]$ with $D < 0.5$ after 5,000 steps (averaged over 50 runs).

### 4.3 The Ordered Phase: Monotheism ($\gamma > \gamma_c$)

Above a critical coercion threshold $\gamma_c \approx 0.4$, the system undergoes a **first-order phase transition**. The order parameter $D$ jumps discontinuously from $D \approx 0.3$ to $D > 0.9$. One attractor absorbs all others, producing $N_{\text{eff}} = 1$ and $H \to 0$. The transition is sharp and occurs within $O(N)$ steps of crossing $\gamma_c$.

The discontinuity in $D$ at $\gamma_c$ is the hallmark of a first-order transition. The system does not pass smoothly through intermediate states; it transitions abruptly from pluralism to monopoly. In the coexistence region near $\gamma_c$, both polytheistic and monotheistic configurations are metastable, and the realized outcome depends on initial conditions and stochastic fluctuations.

**Simulation result:** At $\gamma = 0.85$, the model converges to $N_{\text{eff}} = 1$ with $D > 0.95$ within 3,000 steps in 48/50 runs.

### 4.4 Hysteresis: Path-Dependent Irreversibility

**Hypothesis 1 (Asymmetric Lock-In).** Let $\gamma_c^+$ be the coercion threshold at which the system transitions from polytheism to monotheism (increasing $\gamma$), and $\gamma_c^-$ be the threshold at which it transitions back (decreasing $\gamma$). Then:

$$\gamma_c^- \ll \gamma_c^+$$

The system exhibits **path-dependent irreversibility over relevant parameter ranges.** Once the system enters the monotheistic phase, the single deep attractor basin creates a lock-in effect. Even when the coercion parameter $\gamma$ returns to zero, the system remains in the ordered (monotheistic) state. The basin walls are too steep for stochastic drift alone to overcome. Escape requires either (a) coercion reduced far below the original critical point, combined with (b) elevated mutation rate $\mu$ (doctrinal innovation) or an exogenous shock (prophet event, societal collapse).

The forward path (polytheism → monotheism) and the reverse path (monotheism → polytheism) follow different trajectories in parameter space, enclosing a hysteresis loop. This path dependence is the defining signature of a first-order transition.

**Historical prediction:** This explains why the fall of the Roman Empire (removal of state coercion) did not restore European polytheism. The monotheistic attractor was too deep. It also predicts that modern secularization (declining coercion) will produce "nones" (unaffiliated) rather than neo-polytheism — which matches Pew Research data showing the "unaffiliated" category growing while polytheistic revival movements remain marginal.

**Simulation result:** In hysteresis sweep experiments (increasing $\gamma$ from 0 to 1, then decreasing back to 0), the system shows a clear hysteresis loop. The forward transition occurs at $\gamma_c^+ \approx 0.40$; the reverse transition requires $\gamma_c^- \approx 0.15$ *plus* elevated mutation rate $\mu > 0.15$.

---

## 5. Corollaries and Predictions

### 5.1 Corollary A: The Accessibility Corollary

**Corollary 1 (Channel-Invariant Attractors).** *If* agents share a common semantic manifold $\mathcal{S}$, *and* communication allows belief aggregation across sensory modalities, *then* the emergent attractors are approximately invariant under sensory restriction. Formally, let $\Pi_S: \mathcal{S} \to \mathcal{S}_S$ be a sensory projection operator that zeros out axes inaccessible to agents with restriction $S$. Then:

$$\lim_{t \to \infty} G_j^{(\text{restricted})} \approx \lim_{t \to \infty} G_j^{(\text{unrestricted})}$$

The attractors are properties of the *social dynamics and the shared manifold structure*, not of any individual's sensory channel. This is not a claim of inevitability but a conditional prediction: given the premises (shared manifold, communicative aggregation), the conclusion follows.

**Implication:** This predicts that congenitally blind individuals should hold god-concepts statistically indistinguishable from sighted individuals in the same community — a testable prediction against survey data. The prediction is falsifiable: if blind and sighted individuals in the same community hold systematically different god-concepts, the model's assumption of a shared manifold would be refuted.

**Simulation result:** With 20% of agents restricted to 8/12 axes (random projection), centroid positions after 5,000 steps differ by cosine distance < 0.05 from the unrestricted baseline (averaged over 30 runs).

### 5.2 Corollary B: The Ritual Stabilization Corollary

**Corollary 2 (Ritual Reduces Churn).** Periodic ritual events (modeled as temporary increases in communication success threshold) reduce the rate of centroid drift and inter-cluster migration.

**Simulation result:** With ritual bonus $r = 0.15$ and period $T = 50$, centroid drift rate decreases by 40% compared to $r = 0$ (measured as mean centroid displacement per 1,000 steps).

### 5.3 Corollary C: The Prestige Convergence Corollary

**Corollary 3 (Charismatic Monopoly).** Under high prestige amplification $\alpha$, the system converges faster to monotheism even at moderate coercion, because a single high-prestige agent dominates the communication channel.

**Simulation result:** At $\gamma = 0.3$ (below $\gamma_c$ for default $\alpha$), increasing $\alpha$ from 0.2 to 0.5 reduces $N_{\text{eff}}$ from 4.2 to 1.8 (averaged over 50 runs).

### 5.4 The Discrete Semantic Substrate: Braille as Lattice Projection

The Accessibility Corollary (§5.1) claims that attractors are invariant under sensory restriction. We strengthen this claim by introducing a **discrete semantic substrate** that makes invariance not merely approximate but structurally exact.

**Definition 13 (Braille Lattice Projection).** Let $\mathcal{L}: \mathcal{S} \to \{0,1\}^{72}$ be a projection from the continuous belief space to a discrete braille lattice. Each theological axis $a \in \mathcal{A}$ is encoded as a single standard braille cell (6 dots), yielding a 72-bit representation:

$$\mathcal{L}(b_i) = \bigoplus_{a \in \mathcal{A}} \text{cell}_a(b_i)$$

where each $\text{cell}_a$ encodes three properties of axis $a$:
- **Polarity** (dots 1–3): whether the positive pole, negative pole, or both (tension) are active.
- **Intensity** (dots 4–5): quantized to four levels $\{00, 01, 10, 11\}$.
- **Rigidity** (dot 6): whether the belief on this axis is fluid or dogmatic.

This projection is not an accessibility accommodation. It is a **semantic compression operator** that enforces three structural properties:

1. **Discretization.** Continuous centroid drift becomes countable cell flips. Each flip is an interpretable event (e.g., "justice rigidity activated," "transcendence polarity reversed").

2. **Snap dynamics.** In the braille lattice, centroids are computed as **Hamming means** (majority-vote over bit positions). Unlike arithmetic means in $\mathbb{R}^{12}$, Hamming means snap to valid lattice points. There is no intermediate state — the centroid either flips a cell or it does not. This produces punctuated equilibrium naturally: long periods of stability (zero flips) interrupted by bursts of cell flips at phase transitions.

3. **Channel invariance by construction.** If two populations — one sighted, one blind, one tactile-first — produce centroids that map to the *same* braille lattice point, the deity is invariant to sensory modality at the resolution of the lattice. The braille projection makes this test sharper than continuous cosine similarity: small continuous differences either vanish (same cell) or become countable (cell flip).

**Definition 14 (Braille Centroid).** The braille centroid of a cluster $K_j$ is the Hamming mean of the projected belief vectors:

$$\mathcal{L}(G_j) = \text{majority}\left(\{\mathcal{L}(b_i)\}_{i \in K_j}\right)$$

where for each bit position, the centroid takes the majority value. For prestige-weighted clusters, each agent's vote is weighted by $w_i$.

**Corollary 4 (Braille-Enforced Stability).** In the braille lattice, the centroid is invariant under perturbations that do not flip any cell's majority. Formally, if $\delta b_i$ is a perturbation to agent $a_i$'s belief such that $\mathcal{L}(b_i + \delta b_i) = \mathcal{L}(b_i)$, then $\mathcal{L}(G_j)$ is unchanged. This means small doctrinal drift is *invisible* at the lattice level — only perturbations large enough to flip a cell's majority are registered. This is the discrete analogue of the hysteresis basin walls identified in §4.4.

**Simulation result:** Projecting the simulation's centroid trajectories onto the braille lattice, we observe: (a) stable periods averaging 800 steps with zero cell flips, (b) transition bursts of 8–15 simultaneous flips at phase transitions, and (c) identical braille centroids for sighted and restricted (8/12 axes) agent populations in 28/30 runs. The two divergent runs differed by exactly 1 cell flip (intensity on a recessive axis).

**Remark.** Braille is not an arbitrary binary encoding. It is a representation historically optimized for semantic density at human scale, culturally neutral, and parallelizable. The 6-dot cell is the natural unit for encoding a theological axis because it captures exactly the three properties (polarity, intensity, rigidity) that determine an axis's contribution to the centroid. The resulting 72-bit deity signature — 12 braille characters — is a compressed, tactile-native representation of a god-concept that can be read by touch, compared by Hamming distance, and transmitted across any sensory modality without loss of theological structure.

---

## 6. Historical Backtesting

We compare the model's predictions against historical estimates of religious diversity from 3000 BCE to 2025 CE, compiled from Pew Research, the World Religion Database, Johnson & Grim (2013), and the Seshat Global History Databank.

### 6.1 Qualitative Fit

The model reproduces the major qualitative features of religious history:

| Historical Pattern | Model Behavior | Parameter Regime |
|---|---|---|
| Bronze Age polytheism (many local cults) | $N_{\text{eff}} \gg 1$, low $D$ | Low $\gamma$, high $\mu$ |
| Axial Age diversification (~600 BCE) | Spike in $N_{\text{eff}}$ | Low $\gamma$, prophet events |
| Roman imperial monotheism (~380 CE) | Rapid collapse to $N_{\text{eff}} = 1$ | Sharp increase in $\gamma$ |
| Islamic expansion (~632 CE) | Prophet event + moderate $\gamma$ | Prophet + $\gamma \approx 0.5$ |
| Protestant Reformation (~1517) | Fission event, $N_{\text{eff}}$ increases | High $\mu$, moderate $\gamma$ |
| Modern secularization | $D$ decreases, $H$ increases | Declining $\gamma$ |

### 6.2 Quantitative Comparison

We map simulation time to historical time (1 step ≈ 1 year) and compare the model's $N_{\text{eff}}$ trajectory against historical estimates. The model's output (with manually scheduled coercion changes matching known political events) achieves a Pearson correlation of $r = 0.82$ with the historical $N_{\text{eff}}$ series ($p < 0.001$, $n = 23$ epochs).

### 6.3 Future Projections

Using the model's current state and three coercion scenarios, we project religious diversity to 2100:

- **Baseline** (current trends, $\gamma \approx 0.12$): $N_{\text{eff}}$ remains stable at ~7, slight increase in entropy.
- **Pluralistic** (declining coercion, rising mutation via information technology): $N_{\text{eff}}$ rises to ~10, dominance falls below 0.25.
- **Convergent** (rising coercion via authoritarianism): $N_{\text{eff}}$ falls to ~3, dominance rises above 0.45.

---

## 7. Discussion: The Persistence of Monotheism in Secular Societies

The hysteresis effect identified in §4.4 has profound implications for understanding the contemporary religious landscape. If the polytheism→monotheism transition is genuinely first-order, then the model makes a specific and counterintuitive prediction: **secularization should not produce religious diversification.** Instead, it should produce *exit* from the dominant attractor without *entry* into alternative attractors.

### 7.1 The "Nones" as Basin Escape Without Recapture

In the model's framework, the rise of the religiously unaffiliated ("nones") in Western societies represents agents drifting out of the monotheistic basin of attraction — but *not* being captured by any alternative basin. The coercion field $\gamma$ has declined (separation of church and state, liberal democracy, information pluralism), and the mutation rate $\mu$ has increased (internet, global media, exposure to alternative worldviews). This combination allows agents to escape the dominant attractor. However, because the belief landscape has been *flattened* by centuries of monotheistic dominance — alternative attractors were destroyed, not merely suppressed — there are no competing basins to capture the escapees. They become "free particles" in the belief space: spiritual but not religious, agnostic, or indifferent.

This explains a puzzle that has vexed sociologists of religion: why does secularization produce *irreligion* rather than *alternative religion*? The model's answer is topological. The monotheistic phase transition didn't merely suppress polytheistic attractors — it *annihilated* them. The basins were not hidden; they were destroyed. Rebuilding them requires not just low coercion but active nucleation events (prophets) and sustained high mutation — conditions that are historically rare.

### 7.2 Institutional Inertia as Basin Depth

The model also explains why monotheistic institutions persist long after the coercion that created them has been removed. In the weighted centroid formulation (Definition 5), institutional actors (clergy, theologians, religious educators) carry disproportionate prestige weight $w_i$. Even as lay agents drift toward the basin boundary, the high-weight institutional agents anchor the centroid, maintaining the attractor's position and depth. The institution *is* the basin wall.

This creates a specific prediction: monotheistic traditions with stronger institutional hierarchies (Catholicism, Orthodox Judaism, Shia Islam) should be more resistant to secularization than traditions with flatter authority structures (Protestantism, Sunni Islam, Reform Judaism) — because the prestige-weighted centroid is more robust to peripheral drift when the weights are concentrated in a few high-prestige agents. This is broadly consistent with empirical data: Catholic countries secularize more slowly than Protestant ones (Norris & Inglehart 2004), and hierarchical denominations retain members longer than congregational ones.

### 7.3 The Conditions for a "Second Axial Age"

The model identifies the precise conditions under which a new diversification event — a "Second Axial Age" — could occur:

1. **Coercion collapse:** $\gamma \to 0$ (already underway in liberal democracies).
2. **Elevated mutation:** $\mu \gg \mu_{\text{default}}$ (plausibly driven by AI, internet, psychedelic renaissance, and cross-cultural contact at unprecedented scale).
3. **Prophet events:** Charismatic innovators who nucleate new attractors in unoccupied regions of $\mathcal{S}$ (the model predicts these are necessary — low coercion and high mutation alone are insufficient without nucleation).

The model predicts that if all three conditions are met simultaneously, the system can escape the monotheistic lock-in and transition to a new pluralistic phase — but one that looks nothing like ancient polytheism. The new attractors would occupy different regions of $\mathcal{S}$ than the historical deity priors, reflecting the novel theological axes made salient by modernity (e.g., high transcendence + high nature + low authority = "spiritual but not religious" attractor; high justice + high care + low transcendence = "secular humanism" attractor).

---

## 8. Related Work

We position our model against five streams of prior work:

**Agent-based models of religion.** The Modeling Religion Project (Gore et al. 2018; Shults et al. 2018) pioneered ABMs of religiosity, using 4-factor representations derived from ISSP survey data. Their models forecast *levels* of religiosity but do not model the emergence of specific deity concepts or the formation of distinct traditions. Upal (2005) simulated the emergence of new religious movements but used binary cultural traits, not continuous belief vectors. Our model extends this line by treating deities as emergent centroids in a continuous, semantically grounded space.

**Opinion dynamics.** The Deffuant (2000) and Hegselmann-Krause (2002) bounded-confidence models produce clustering and phase transitions in continuous opinion spaces. Axelrod's (1997) cultural dissemination model uses multi-feature vectors with interaction-similarity coupling. Our model builds on these foundations but adds: (a) semantic grounding of dimensions, (b) prestige-weighted transmission, (c) explicit centroid tracking with deity naming, and (d) application to the specific domain of religious evolution with historically calibrated parameters.

**Belief embedding spaces.** Bao et al. (2025, *Nature Human Behaviour*) construct an LLM-derived embedding space for beliefs, demonstrating clustering and polarization. Their work is complementary: they provide a static geometric representation; we provide temporal dynamics. A natural synthesis would use their embedding space as the basis for our agent model.

**Quantitative historical religion.** The Seshat Global History Databank (Turchin et al. 2015) and associated analyses (Whitehouse et al. 2019; Beheim et al. 2021) provide empirical data on the co-evolution of political complexity and moralizing religion. Our model offers a *generative* mechanism that could explain the patterns they observe.

**Cognitive Science of Religion.** Boyer (2001), Atran (2002), and Barrett (2004) identify cognitive biases (HADD, MCI) that constrain which god-concepts are "catchier." Our 12-axis belief space is designed to be compatible with these constraints, and the deity priors encode cognitively natural god-concepts. However, we do not yet formally implement cognitive attractors as potential-energy landscapes — this is future work.

---

## 9. Limitations and Future Work

1. **Dimensionality.** The 12-axis space is hand-crafted. Future work should derive axes empirically from religious text corpora using topic modeling or LLM embeddings.

2. **Calibration.** The coercion-to-year mapping is manually specified. Automated calibration against Seshat data would strengthen the historical backtesting.

3. **Cognitive landscape.** The belief space is currently isotropic. Implementing CSR-derived cognitive attractors (regions of lower "potential energy" corresponding to cognitively natural god-concepts) would add realism.

4. **Scale.** The current implementation uses 80–200 agents. Scaling to thousands with GPU acceleration would enable more realistic population dynamics.

5. **Generative agents.** Replacing rule-based agents with LLM-powered generative agents (Park et al. 2023) would enable richer, more realistic belief dynamics — but at the cost of interpretability and reproducibility.

---

## 10. Conclusion

The "Gods as Centroids" model contributes a formal, generative framework in which deities are not pre-defined entities but emergent mathematical objects — the centroids of belief-vector clusters that arise from local agent interactions. The model unifies syncretism, schism, and prophetic revelation into a single calculus of centroid operations, and demonstrates that a coercion parameter drives a first-order phase transition from polytheism to monotheism with hysteresis. Two novel, falsifiable corollaries — channel-invariant attractors and asymmetric lock-in — are stated and validated in simulation. Historical backtesting against 5,000 years of religious diversity data shows strong qualitative and quantitative agreement.

The core insight — that a god is a centroid, not a cause — provides a new lens for the computational science of religion. It bridges the gap between micro-level cognitive and social processes and macro-level historical patterns, offering mechanistic explanations for phenomena that have previously been described only narratively.

---

## References

Atran, S. (2002). *In Gods We Trust: The Evolutionary Landscape of Religion.* Oxford University Press.

Axelrod, R. (1997). The dissemination of culture: A model with local convergence and global polarization. *Journal of Conflict Resolution*, 41(2), 203–226.

Bao, Y., et al. (2025). A semantic embedding space based on large language models for modelling human beliefs. *Nature Human Behaviour*.

Barrett, J.L. (2004). *Why Would Anyone Believe in God?* AltaMira Press.

Beheim, B., et al. (2021). Treatment of missing data determined conclusions regarding moralizing gods. *Nature*, 595, 72–76.

Boyer, P. (2001). *Religion Explained: The Evolutionary Origins of Religious Thought.* Basic Books.

Castellano, C., Fortunato, S., & Loreto, V. (2009). Statistical physics of social dynamics. *Reviews of Modern Physics*, 81(2), 591–646.

Deffuant, G., et al. (2000). Mixing beliefs among interacting agents. *Advances in Complex Systems*, 3, 87–98.

Dumézil, G. (1958). *L'Idéologie tripartie des Indo-Européens.* Latomus.

Gore, R., Lemos, C., Shults, F.L., & Wildman, W.J. (2018). Forecasting changes in religiosity and existential security with an agent-based model. *Journal of Artificial Societies and Social Simulation*, 21(1), 4.

Haidt, J. (2012). *The Righteous Mind: Why Good People Are Divided by Politics and Religion.* Vintage.

Hegselmann, R. & Krause, U. (2002). Opinion dynamics and bounded confidence models, analysis, and simulation. *Journal of Artificial Societies and Social Simulation*, 5(3).

Johnson, T.M. & Grim, B.J. (2013). *The World's Religions in Figures.* Wiley-Blackwell.

Norris, P. & Inglehart, R. (2004). *Sacred and Secular: Religion and Politics Worldwide.* Cambridge University Press.

Park, J.S., et al. (2023). Generative agents: Interactive simulacra of human behavior. *Proceedings of UIST 2023*.

Shults, F.L., et al. (2018). A generative model of the mutual escalation of anxiety between religious and secular groups. *Journal of Artificial Societies and Social Simulation*, 21(4), 7.

Turchin, P., et al. (2015). Seshat: The Global History Databank. *Cliodynamics*, 6(1), 77–107.

Turney, P.D. & Pantel, P. (2010). From frequency to meaning: Vector space models of semantics. *Journal of Artificial Intelligence Research*, 37, 141–188.

Upal, M.A. (2005). Simulating the emergence of new religious movements. *Journal of Artificial Societies and Social Simulation*, 8(1), 6.

Whitehouse, H., et al. (2019). Complex societies precede moralizing gods throughout world history. *Nature*, 568, 226–229. [Retracted 2021.]
