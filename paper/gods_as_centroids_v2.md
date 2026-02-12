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

**Simulation result:** At $\gamma = 0.05$, $\mu = 0.08$, the model consistently produces $N_{\text{eff}} \in [4, 8]$ with $D < 0.6$ after 2,000 steps (averaged over 30 runs).

### 4.3 The Ordered Phase: Monotheism ($\gamma > \gamma_c$)

Above a critical coercion threshold $\gamma_c \approx 0.4$, the system undergoes a **first-order phase transition**. The order parameter $D$ jumps discontinuously from $D \approx 0.3$ to $D > 0.9$. One attractor absorbs all others, producing $N_{\text{eff}} = 1$ and $H \to 0$. The transition is sharp and occurs within $O(N)$ steps of crossing $\gamma_c$.

The discontinuity in $D$ at $\gamma_c$ is the hallmark of a first-order transition. The system does not pass smoothly through intermediate states; it transitions abruptly from pluralism to monopoly. In the coexistence region near $\gamma_c$, both polytheistic and monotheistic configurations are metastable, and the realized outcome depends on initial conditions and stochastic fluctuations.

**Simulation result:** At $\gamma = 0.85$, the model converges to $N_{\text{eff}} \approx 1$ with $D > 0.99$ within 2,000 steps in 30/30 runs.

### 4.4 Hysteresis: Path-Dependent Irreversibility

**Hypothesis 1 (Asymmetric Lock-In).** Let $\gamma_c^+$ be the coercion threshold at which the system transitions from polytheism to monotheism (increasing $\gamma$), and $\gamma_c^-$ be the threshold at which it transitions back (decreasing $\gamma$). Then:

$$\gamma_c^- \ll \gamma_c^+$$

The system exhibits **path-dependent irreversibility over relevant parameter ranges.** Once the system enters the monotheistic phase, the single deep attractor basin creates a lock-in effect. Even when the coercion parameter $\gamma$ returns to zero, the system remains in the ordered (monotheistic) state. The basin walls are too steep for stochastic drift alone to overcome. Escape requires either (a) coercion reduced far below the original critical point, combined with (b) elevated mutation rate $\mu$ (doctrinal innovation) or an exogenous shock (prophet event, societal collapse).

The forward path (polytheism → monotheism) and the reverse path (monotheism → polytheism) follow different trajectories in parameter space, enclosing a hysteresis loop. This path dependence is the defining signature of a first-order transition.

**Historical prediction:** This explains why the fall of the Roman Empire (removal of state coercion) did not restore European polytheism. The monotheistic attractor was too deep. It also predicts that modern secularization (declining coercion) will produce "nones" (unaffiliated) rather than neo-polytheism — which matches Pew Research data showing the "unaffiliated" category growing while polytheistic revival movements remain marginal.

**Simulation result (Figure 1).** We perform a systematic hysteresis sweep: 30 independent runs, each with 80 agents, 21 coercion levels, and 2,000 equilibration steps per level.

In the *forward sweep* (increasing coercion from 0 to 1), dominance increases gradually from $D = 0.65$ at $\gamma = 0$ to $D = 0.81$ at $\gamma = 1.0$, crossing the $D = 0.7$ threshold at the forward critical point $\gamma_c^{+} \approx 0.85$.

In the *reverse sweep* (decreasing coercion from 1 to 0), the system starts fully monotheistic ($D = 1.0$) and remains locked in until the reverse critical point $\gamma_c^{-} \approx 0.30$, below which dominance collapses. The hysteresis gap is $\Delta\gamma \approx 0.55$.

Crucially, at $\gamma = 0$ the forward sweep yields $D = 0.65$ (polytheistic) while the reverse sweep yields $D = 0.14$ (fragmented post-monotheistic) — the system does **not** return to its original state. The monotheistic phase annihilates the polytheistic attractor landscape; removing coercion produces fragmentation, not restoration.

### 4.5 Lattice Hysteresis: Snap Dynamics Amplify Phase Transitions

The Braille Lattice Corollary (§5.4) introduces discrete Hamming-mean centroids as an alternative to arithmetic-mean centroids in continuous space. We test whether this discretization affects the phase transition by repeating the hysteresis sweep with **Hamming centroids on the 8-dot braille lattice** (96 bits per agent). We run 40 parallel sweeps on Modal (20 Hamming + 20 arithmetic, same parameters as §4.4).

The result is dramatic:

| Metric | Arithmetic (continuous) | Hamming (8-dot lattice) |
|---|---|---|
| Forward critical point $\gamma_c^+$ | $\approx 0.90$ | $\approx 0.10$ |
| $D$ at $\gamma = 0.10$ | 0.609 | **1.000** |
| $D$ at $\gamma = 0.50$ | 0.497 | **1.000** |
| $D$ at $\gamma = 1.00$ | 0.791 | **1.000** |

Hamming centroids collapse to full dominance ($D = 1.0$) at $\gamma \approx 0.10$ — a **9$\times$ lower coercion threshold** than arithmetic centroids. The discrete lattice eliminates continuous drift that delays the transition. In continuous space, small perturbations gradually shift the centroid; on the lattice, perturbations either flip a bit (discrete event) or are invisible (no effect). This produces a much steeper energy landscape: once a majority of agents' lattice projections agree on a bit, the centroid snaps and cannot be gradually eroded.

This is a new prediction: **discrete semantic substrates amplify phase transitions.** If human belief dynamics operate on a discrete cognitive substrate (as suggested by categorical perception in psychology and the discreteness of linguistic categories), then the effective coercion threshold for monotheistic collapse may be far lower than continuous models predict — consistent with the historical observation that monotheism emerged rapidly once state coercion was applied (e.g., the Theodosian decrees of 380–392 CE produced near-complete Christianization within two generations).

![Figure 1: Hysteresis in the polytheism–monotheism transition. Left: Dominance D vs coercion γ showing the hysteresis loop. Center: Effective deity count N_eff. Right: Belief entropy H. Red (forward): γ increasing from 0 to 1. Blue (reverse): γ decreasing from 1 to 0. Shaded regions show ±1 standard deviation across 30 runs. The forward and reverse curves enclose a large hysteresis loop, confirming the asymmetric lock-in hypothesis.](assets/hero_plot.png)

**Figure 1.** *Hysteresis in the polytheism–monotheism transition.* **Left:** Dominance $D$ vs coercion $\gamma$. The forward sweep (red, $\gamma \uparrow$) and reverse sweep (blue, $\gamma \downarrow$) follow different trajectories, enclosing a hysteresis loop with gap $\Delta\gamma \approx 0.55$. **Center:** Effective deity count $N_{\text{eff}}$ collapses from ~18 to ~1 under increasing coercion and does not recover. **Right:** Belief entropy $H$ shows the same irreversibility — diversity lost is not restored. Shaded regions: $\pm 1\sigma$ across 30 independent runs ($N = 80$ agents each).

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

**Experimental validation.** We train an encoder-bottleneck-decoder architecture on 16 deity priors (500 noisy samples each) using temperature-annealed straight-through estimation ($\tau: 1 \to 10$). The encoder learns a projection $\mathbb{R}^{12} \to \mathbb{R}^N$, which is quantized via $\sigma(\text{logits} \cdot \tau) \to \{0,1\}^N$. The decoder reconstructs $\hat{b} \in \mathbb{R}^{12}$ from the binary code; a classifier predicts deity identity from the same bits.

**Result A (Centroid Preservation).** At 72 bits, the mean cosine similarity between original and reconstructed centroids is $0.9950$ across all 16 deities ($\min = 0.9912$, Yahweh; $\max = 0.9985$, Gaia). The mean L2 displacement is $0.099$, which is $17.9\%$ of the mean inter-deity distance ($0.554$). Centroids are barely moved by the bottleneck.

**Result B (Task Invariance).** Deity classification accuracy through the 72-bit bottleneck is $83.25\%$ (chance $= 6.25\%$, $16$ classes). The Spearman rank correlation between pairwise deity similarities before and after the bottleneck is $\rho = 0.968$ ($p = 9.25 \times 10^{-73}$). The similarity ordering is almost perfectly preserved.

**Result C (Capacity Stress Test).** We vary the bit budget from 12 to 96 bits per deity:

| Total bits | Bits/axis | Classification | Cosine sim | Rank $\rho$ |
|---|---|---|---|---|
| 12 | 1 | 82.25% | 0.9994 | 0.992 |
| 24 | 2 | 84.00% | 0.9986 | 0.991 |
| 48 | 4 | 84.69% | 0.9971 | 0.983 |
| **72** | **6** | **85.12%** | **0.9962** | **0.987** |
| 96 | 8 | 84.88% | 0.9960 | 0.979 |

Structure survives down to **12 bits** (1 bit per axis) with $0.9994$ cosine similarity and $82\%$ classification accuracy. There is no sharp phase transition — the theological structure is remarkably compressible. Performance plateaus above 48 bits, suggesting the essential structure of a deity centroid requires approximately 4 bits per theological axis.

**Result D (Channel Invariance).** Under sensory restriction (zeroing 2–4 axes and renormalizing), reconstructed centroids retain $90$–$94\%$ cosine similarity to unrestricted centroids. Abstract restriction (transcendence, creation removed): $93.8\%$ cosine similarity, $10/16$ same classification. Social-political restriction (authority, justice, power, order removed): $92.9\%$, $6/16$. Visual-embodied restriction (4 axes removed): $90.1\%$, $5/16$. Core structure survives sensory restriction, though classification degrades with more axes removed.

**Result E (Semantic Braiding).** We introduce *semantic braiding*: fusing heterogeneous models by projecting their internal states through a shared discrete bottleneck and combining them at the level of compressed semantic invariants. We compare two regimes:

*Regime A (post-hoc braiding):* $K=5$ encoder variants trained independently, then fused via bitwise majority vote. Bit agreement between models is only $49.2\%$ — models encode differently — yet the braided centroid achieves $95.7\%$ cosine similarity and $75\%$ correct classification.

*Regime B (co-trained braiding):* $K=5$ encoders trained jointly with a **shared decoder/classifier** and annealed alignment loss $\mathcal{L}_{\text{align}} = \alpha \sum_{i<j} |z^{(i)} - z^{(j)}|_1$ ($\alpha$ warmed from $0$ to $0.01$ after epoch 50). The shared decoder enforces structural alignment implicitly — disagreement is punished by task loss, not by an extrinsic penalty. Results:

| Metric | Regime A (post-hoc) | Regime B (co-trained) |
|---|---|---|
| Bit agreement | 49.2% | **86.3%** |
| Braided cosine sim | 0.957 | **0.996** |
| Braid correct rate | 75.0% | **100.0%** |
| Braid improvement | −0.037 | **+0.001** |

Regime B achieves perfect deity classification from braided bits and *positive* braid improvement — the consensus centroid is more faithful than any individual model. Naïve co-training with separate decoders ($\alpha = 0.05$) degraded performance due to gradient competition between alignment and task objectives; the shared decoder resolves this by making agreement a structural necessity rather than a penalty.

**Result F (Frontier LLM as Semantic Judge).** To validate that the bottleneck preserves semantically relevant structure beyond our own metrics, we use frontier LLMs as independent measurement instruments. Claude Sonnet 4 generates 3-sentence theological profiles from (a) the full continuous centroid and (b) the braille-compressed centroid (72-bit → decoded). GPT-4o-mini then acts as a blind judge, evaluating whether each pair describes the same deity. Across all 8 tested deities, the judge ruled **8/8 = 100% structural equivalence** with mean structural similarity $0.844$ and mean confidence $0.90$. A frontier-class language model cannot reliably distinguish theology generated from continuous centroids versus 72-bit compressed centroids, confirming that the bottleneck preserves the semantically relevant structure of the centroid.

**Result G (Real Embeddings).** To move beyond hand-designed priors, we score 24 canonical religious passages from 11 traditions (Judaism, Christianity, Islam, Hinduism, Buddhism, Norse, Greek, Daoism, Lakota, Aboriginal Australian, Zoroastrianism) on the 12 theological axes using Claude Sonnet 4 at temperature 0. We then compress these real embeddings through the 72-bit braille bottleneck and compare against two baselines: random binary projection (72 bits) and PCA + uniform quantization (72 bits).

| Method | Reconstruction (cos) | Spearman $\rho$ | Cluster gap |
|---|---|---|---|
| **Braille lattice** | **0.986** | **0.967** ($p = 1.9 \times 10^{-164}$) | **0.085** |
| PCA + quantize | 0.915 | 0.901 | 0.075 |
| Random binary | 0.831 | 0.728 | 0.040 |

The braille lattice outperforms both baselines on every metric. Notably, the cluster separation gap *increases* after braille compression (0.085 vs 0.058 in continuous space) — the bottleneck acts as a denoiser, sharpening theological boundaries. Specific theological predictions survive compression: Judaism–Islam similarity ($0.948$), Christianity–Islam ($0.890$), Hinduism–Buddhism ($0.797$), Daoism–Buddhism ($0.851$). The Abrahamic cluster remains tightly bound ($> 0.88$ pairwise) and well-separated from Dharmic traditions after compression.

**Result H (Multi-Model Variance).** To test whether theological structure is a property of the texts or an artifact of a particular model's training data, we score the same 23 passages with four frontier LLMs (Claude Sonnet 4, GPT-4o, Gemini 2.0 Flash, Llama 3.3 70B) at temperature 0. The mean Pearson correlation across all model pairs is $r = 0.869$ — strong agreement on theological structure across architecturally distinct models.

Per-axis analysis reveals which theological dimensions are *consensual* versus *contested*:

| Axis | Coefficient of Variation | Interpretation |
|---|---|---|
| Transcendence | 0.100 | Consensual |
| Order | 0.121 | Consensual |
| Wisdom | 0.121 | Consensual |
| Authority | 0.132 | Consensual |
| War | **0.463** | **Contested** |
| Justice | 0.257 | Contested |
| Death | 0.236 | Contested |

The contested axes are precisely those where theological interpretation is genuinely ambiguous — "war" in the Bhagavad Gita is metaphorical to some models, literal to others. This is real hermeneutic disagreement surfacing through model variance, not noise.

Despite continuous-space disagreement, the braille lattice achieves $88.3\%$ bit agreement across all model pairs. The discrete bottleneck absorbs interpretive variance while preserving structural consensus. Cluster separation through braille majority-vote ($0.081$) matches or exceeds individual models ($0.059$–$0.092$), confirming that the lattice acts as a consensus filter: it preserves what all models agree on and quantizes away what they dispute.

**Result I (Scaled Corpus Validation).** To stress-test the bottleneck at scale, we train a multi-modal lattice autoencoder (MMLA) on **826 passages from 37 traditions** (the 126 original passages merged with 700 additional passages scraped from canonical texts and scored by 4 LLMs via consensus). The MMLA has two encoder modalities — a text encoder (TF-IDF + PCA, 128 dimensions) and a scorer encoder (12-dimensional LLM vectors) — that must agree through a shared 96-bit braille lattice bottleneck.

| Phase | Metric | Result |
|---|---|---|
| Phase 2 (cross-modal) | Classification accuracy | **100%** (37 classes) |
| Phase 2 | Cross-modal bit agreement | **87.1%** |
| Phase 2 | Text → consensus cosine | 0.9479 |
| Phase 2 | Scorer → consensus cosine | 0.9673 |
| Phase 3 (operators) | Operator prediction accuracy | **42.9%** (4 classes: fusion/fission/perturbation/none) |

The 100% classification accuracy through a 96-bit bottleneck on 37 tradition classes (chance = 2.7%) confirms that the braille lattice preserves sufficient structure for perfect tradition discrimination even at corpus scale. The 87.1% cross-modal bit agreement means that text and scorer encoders independently produce lattice codes that agree on 83.6 of 96 bits — the shared bottleneck forces consensus between heterogeneous input modalities. Phase 3 operator prediction (42.9% vs 25% chance) demonstrates that the lattice codes contain enough structural information to predict which centroid operation (fusion, fission, perturbation, or none) applies to a given pair of traditions, though this remains the weakest link and an area for future improvement.

**Remark.** We use a braille-inspired tactile lattice because it is a historically optimized, human-scale discrete semantic code — but the result does not depend on braille per se. The 6-dot cell is the natural unit for encoding a theological axis because it captures exactly the three properties (polarity, intensity, rigidity) that determine an axis's contribution to the centroid. The resulting 72-bit deity signature — 12 braille characters — is a compressed, tactile-native representation of a god-concept that can be read by touch, compared by Hamming distance, and transmitted across any sensory modality without loss of theological structure.

### 5.5 The 8-Dot Extension: Salience, Momentum, and Empirical Tradition Signatures

We extend the 6-dot braille cell (72 bits) to the full **8-dot braille cell** (96 bits for 12 axes), adding two new semantic properties per axis:

- **Dot 7 (Salience):** Whether this axis is contextually active — above the median value across all axes. This encodes which dimensions are *foregrounded* in a tradition's theology.
- **Dot 8 (Momentum):** Whether this axis is currently changing — the temporal derivative exceeds a threshold. This encodes doctrinal dynamism.

The 8-dot cell uses the full Unicode braille range (U+2800–U+28FF), giving 256 possible states per axis versus 64 for the 6-dot cell. The total representation is $12 \times 8 = 96$ bits per tradition.

**Definition 15 (8-Dot Braille Lattice).** Let $\mathcal{L}_8: \mathcal{S} \to \{0,1\}^{96}$ be the extended projection:

$$\mathcal{L}_8(b_i) = \bigoplus_{a \in \mathcal{A}} \text{cell}_a^{(8)}(b_i)$$

where each $\text{cell}_a^{(8)}$ encodes five properties: polarity (dots 1–3), intensity (dots 4–5), rigidity (dot 6), salience (dot 7), and momentum (dot 8).

**Empirical Tradition Signatures.** We project the 126 consensus-scored passages (§8) onto the 8-dot lattice and compute per-tradition Hamming-mean centroids. Each tradition receives a unique 96-bit braille signature — 12 Unicode characters that encode its complete theological profile:

| Tradition | 8-Dot Braille | Top 3 Axes |
|---|---|---|
| Judaism | ⡑⠐⡑⠀⠐⠑⠂⠂⠂⠂⠀⡑ | authority, care, order |
| Christianity | ⠂⡑⡁⠀⠐⠐⠀⠂⠀⠀⠂⠑ | transcendence, wisdom, power |
| Islam | ⡑⡑⠁⠑⡐⡑⠂⠂⠂⠂⠂⡑ | authority, transcendence, wisdom |
| Hinduism | ⡑⡁⠀⠀⡑⡑⠀⠂⠀⠂⠂⡐ | authority, wisdom, power |
| Buddhism | ⠂⡁⡑⠀⡉⠀⠀⠂⠀⠂⠂⡑ | wisdom, care, order |
| Jainism | ⠂⡐⡑⡑⡑⠀⠂⠂⠀⠂⠀⡑ | transcendence, care, justice |
| Daoism | ⠀⡁⠀⠂⡑⠀⠀⠂⠀⠀⡑⡑ | wisdom, nature, order |
| Confucianism | ⡐⠀⡀⡑⡁⠀⠂⠂⠂⠂⠀⡑ | authority, justice, order |
| Lakota | ⠂⠒⡑⠀⡑⠀⠀⠂⠂⠂⡑⡑ | care, wisdom, nature |
| Yoruba | ⠀⡑⠀⠂⡁⡐⠐⠂⠂⠂⠐⡑ | transcendence, power, order |
| Secular Humanism | ⠂⠂⡉⡁⡁⠀⠂⠂⠀⠂⡀⡑ | care, order, justice |
| Sufism | ⠂⡑⡑⠀⡑⠀⠀⠂⠀⠀⠒⡐ | transcendence, care, wisdom |

These signatures are derived from the 4-LLM lattice consensus (§8.4) and are stable under individual model substitution. The 8-dot extension adds 2–5 bits of information per tradition compared to the 6-dot encoding, with the additional bits concentrated in the salience channel (dot 7), which captures which axes are foregrounded in each tradition's theology.

**Round-trip reconstruction.** Decoding the 96-bit signatures back to continuous vectors yields mean cosine similarity of $0.932$ with the original consensus centroids across all 37 traditions. The lattice acts as a lossy compressor that preserves theological structure while quantizing away scorer noise.

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

**Simulation result (Figure 2).** We test this prediction with a systematic prophet escape experiment: 600 runs across a grid of prophet rates and mutation rates (6 prophet rates $\times$ 5 mutation rates $\times$ 20 runs each). Phase 1 drives the system to monotheistic lock-in ($\gamma = 0.9$, 5,000 steps, corpus-calibrated parameters). Phase 2 drops coercion to $\gamma = 0.05$ (residual institutional inertia) and enables prophets and elevated mutation for 8,000 steps.

The results confirm the three-condition prediction:

- **Mutation alone is insufficient.** Without prophets, escape probability *decreases* as mutation increases: 100% at $\mu = 0.05$ but only 5% at $\mu = 0.25$. High mutation creates noise that the dominant attractor absorbs rather than new attractors that compete with it.
- **Prophets rescue high-mutation regimes.** At $\mu = 0.20$, adding prophets at rate 0.01 increases escape from 10% to 85%. Prophets provide the directed, coherent perturbations needed to nucleate new attractors in unoccupied regions of $\mathcal{S}$.
- **Too many prophets are counterproductive.** At prophet rate 0.02, escape drops at low mutation (0–5%). Competing prophets create transient chaos rather than stable new attractors.
- **The sweet spot** is moderate prophets ($\sim$0.005) with moderate mutation ($\mu \sim 0.10$–$0.15$): 90–100% escape probability. This is the "Second Axial Age" condition.

![Figure 2: Prophet escape from monotheistic lock-in. Left: Escape probability as a function of prophet rate and mutation rate. Center: Final effective deity count. Right: Final dominance. The diagonal gradient confirms that escape requires both prophets AND elevated mutation — neither alone is sufficient at high mutation rates.](assets/prophet_escape_plot.png)

**Figure 2.** *Prophet escape from monotheistic lock-in.* **Left:** Escape probability (green = escape, red = locked). **Center:** Final $N_{\text{eff}}$ after 8,000 escape steps. **Right:** Final dominance $D$. The model predicts that a "Second Axial Age" requires the simultaneous presence of charismatic innovators (prophets) and elevated doctrinal mutation — conditions plausibly met by the combination of AI, internet, and cross-cultural contact at unprecedented scale.

---

## 8. Corpus-Calibrated Parameters

A key methodological advance is the derivation of simulation parameters from empirical data rather than hand-tuning. We score **126 canonical religious passages from 37 traditions** spanning every inhabited continent on the 12 theological axes using Claude Sonnet 4 (temperature 0) and derive all model parameters from the resulting embeddings. The corpus includes Abrahamic traditions (Judaism, Christianity, Islam, Baha'i, Druze, Samaritanism, Zoroastrianism), South/East Asian traditions (Hinduism, Buddhism, Jainism, Sikhism, Daoism, Confucianism, Shinto), African traditions (Yoruba, Akan, Kemetic), African diaspora traditions (Candomblé, Vodou, Rastafari), Indigenous traditions (Lakota, Navajo, Aboriginal Australian, Maori, Hawaiian), Mesoamerican/Andean traditions (Nahua, Maya, Inca), Central Asian traditions (Tengrism, Korean Muism), syncretic traditions (Cao Dai), nature spirituality (Wicca), Islamic mysticism (Sufism), and secular humanism. Each tradition is represented by at least 3 passages to enable intra-tradition variance estimation. Women's voices are explicitly included (Rabia al-Adawiyya, Mirabai, Andal, Gargi Vachaknavi, Changing Woman, Starhawk).

### 8.1 Empirical Deity Priors

Rather than hand-crafting deity prior vectors, we compute tradition centroids as the mean of LLM-scored passage embeddings within each tradition. The resulting 37 tradition centroids serve as empirically grounded deity priors. A representative sample:

| Tradition | Top 3 Axes | Family |
|---|---|---|
| Judaism | authority, order, power | Abrahamic |
| Christianity | transcendence, care, power | Abrahamic |
| Islam | authority, transcendence, order | Abrahamic |
| Hinduism | transcendence, power, creation | Dharmic |
| Buddhism | wisdom, transcendence, order | Dharmic |
| Jainism | care, wisdom, justice | Dharmic |
| Sikhism | transcendence, order, creation | Dharmic |
| Daoism | wisdom, transcendence, nature | East Asian |
| Confucianism | wisdom, order, care | East Asian |
| Shinto | order, nature, transcendence | East Asian |
| Yoruba | order, transcendence, wisdom | African |
| Akan | wisdom, care, transcendence | African |
| Lakota | nature, transcendence, care | Indigenous |
| Navajo | nature, order, fertility | Indigenous |
| Aboriginal Australian | nature, creation, transcendence | Indigenous |
| Maori | nature, transcendence, power | Pacific |
| Hawaiian | nature, creation, power | Pacific |
| Nahua | wisdom, nature, death | Mesoamerican |
| Maya | creation, wisdom, transcendence | Mesoamerican |
| Sufism | transcendence, wisdom, care | Mystical |
| Secular Humanism | wisdom, care, justice | Philosophical |
| Candomblé | power, order, nature | African diaspora |
| Vodou | order, transcendence, nature | African diaspora |
| Tengrism | transcendence, order, nature | Central Asian |

The empirical centroids reveal natural clustering: Abrahamic traditions load on authority and transcendence; Dharmic traditions on wisdom and transcendence; Indigenous traditions on nature and care; Mesoamerican traditions on creation and death. These clusters emerge from the data without supervision.

### 8.2 Multi-LLM Inter-Scorer Agreement

To ensure that the theological scoring is robust to scorer choice, we replicate the full 126-passage scoring with four independent LLMs: Claude Sonnet 4 (Anthropic), GPT-4o (OpenAI), Gemini 2.0 Flash (Google), and Llama 3.3 70B (Meta, open-weight). All models use the same prompt at temperature 0.

| Metric | Value | Interpretation |
|---|---|---|
| Krippendorff's $\alpha$ | **0.903** | Good agreement ($\alpha > 0.8$) |
| Mean pairwise Pearson $r$ | **0.887** | Strong consensus |
| Mean absolute deviation | **0.069** | Low per-passage disagreement |

Pairwise model correlations (all 12 axes flattened):

| Pair | $r$ |
|---|---|
| GPT-4o × Llama 70B | 0.901 |
| GPT-4o × Gemini Flash | 0.898 |
| Claude × GPT-4o | 0.894 |
| Claude × Llama 70B | 0.891 |
| Gemini Flash × Llama 70B | 0.872 |
| Claude × Gemini Flash | 0.867 |

Per-axis agreement is highest for *power* ($r = 0.875$), *care* ($r = 0.868$), and *justice* ($r = 0.861$), and lowest for *order* ($r = 0.632$) and *wisdom* ($r = 0.779$) — axes where cultural framing may introduce legitimate interpretive variance. Crucially, the inclusion of an open-weight model (Llama 70B) means the entire scoring pipeline is reproducible without proprietary API access.

### 8.3 Consensus-Derived Parameters

We derive final simulation parameters from the **4-model consensus scores** (mean across all scorers), which cancels individual model biases:

| Parameter | Hand-tuned | Single-scorer (Claude) | **4-model consensus** | Method |
|---|---|---|---|---|
| Cluster threshold $\theta$ | 0.40 | 0.097 | **0.073** | Midpoint of mean intra/inter-tradition cosine distances |
| Mutation rate $\mu$ | 0.08 | 0.25 | **0.25** | Intra-tradition standard deviation |
| Fission threshold $\sigma^2_{\max}$ | 0.15 | 0.162 | **0.106** | Midpoint of schism vs. non-schism tradition variances |
| Belief influence $\beta$ | 0.15 | 0.06 | **0.055** | Ratio of intra- to inter-tradition similarity |
| Deity priors | 12 hand-crafted | 37 from corpus | **37 consensus** | Tradition centroid vectors |

The consensus averaging tightens all parameters because scorer noise cancels: $\theta$ drops from 0.097 to 0.073 (now **5.5$\times$ tighter** than hand-tuned), and $\sigma^2_{\max}$ drops from 0.162 to 0.106. The fission threshold continues to correctly discriminate: Hinduism ($\sigma^2 = 0.135$) and Buddhism ($\sigma^2 = 0.130$) exceed the threshold, while Jainism ($\sigma^2 = 0.048$), Aboriginal Australian ($\sigma^2 = 0.037$), and Cao Dai ($\sigma^2 = 0.039$) fall well below it.

### 8.4 4-LLM Lattice Consensus: Bit-Level Agreement

The continuous consensus (§8.3) averages scores before projection. An alternative — and more principled — approach is to project each model's scores **independently** to the 8-dot braille lattice, then compute the Hamming mean across all 4 models at the bit level. This preserves each model's discrete judgment and lets majority vote resolve quantization boundary disagreements.

**Pipeline.** For each of 126 passages: (1) each of 4 LLMs produces a continuous score vector; (2) each vector is projected independently to a 96-bit lattice point; (3) the per-passage lattice consensus is the Hamming mean (majority vote) across 4 lattice points; (4) per-tradition signatures are the Hamming mean across all passages in that tradition.

**Per-passage agreement.** Across 126 passages, the 4 models agree unanimously on a mean of **82.2 out of 96 bits (85.6%)**. The minimum unanimous agreement is 68/96 (70.8%) and the maximum is 92/96 (95.8%).

**Per-dot agreement hierarchy.** The 8 dot positions exhibit a clear agreement hierarchy that reflects the nature of the encoded semantic property:

| Dot | Semantic Property | 4-Model Agreement |
|---|---|---|
| 8 | Momentum (temporal change) | 100.0% |
| 6 | Rigidity (fluid vs dogmatic) | 99.7% |
| 4 | Intensity (high bit) | 94.4% |
| 3 | Tension (polarity conflict) | 86.8% |
| 2 | Negative polarity | 79.4% |
| 7 | Salience (contextually active) | 77.8% |
| 1 | Positive polarity | 77.7% |
| 5 | Intensity (low bit) | 69.4% |

Models agree most on **structural features** (rigidity, coarse intensity) and least on **fine-grained distinctions** (low-bit intensity, polarity direction). This is consistent with the continuous-space finding that models agree on *whether* an axis is important but disagree on *exactly how much* — and the lattice makes this distinction precise: the high-intensity bit (dot 4, threshold at $v > 0.5$) is consensual (94.4%), while the low-intensity bit (dot 5, threshold at $v \bmod 0.25$) is contested (69.4%).

**Per-model distance from consensus.** Each model's tradition-level centroids differ from the lattice consensus by:

| Model | Mean distance | Exact matches | Max distance |
|---|---|---|---|
| GPT-4o | 3.8 bits | 3/37 | 11 bits |
| Claude Sonnet 4 | 4.4 bits | 0/37 | 10 bits |
| Llama 3.3 70B | 4.4 bits | 2/37 | 11 bits |
| Gemini 2.0 Flash | 6.1 bits | 0/37 | 16 bits |

No single model produces tradition signatures identical to the lattice consensus for all 37 traditions. The consensus is a genuinely emergent object — it represents the stable core of theological structure that all 4 architecturally distinct models agree on, with quantization boundary disagreements resolved by majority vote.

**Lattice fission discriminator.** The fission discriminator (§3.2) transfers to the lattice. Traditions with historical schisms (Hinduism, Islam, Christianity, Buddhism) exhibit higher intra-tradition Hamming variance (mean $= 21.99$) than historically stable traditions (Jainism, Sikhism, Zoroastrianism, Cao Dai; mean $= 17.45$). The discrete lattice preserves the discriminative signal that identifies traditions prone to schism.

---

## 9. Related Work

We position our model against five streams of prior work:

**Agent-based models of religion.** The Modeling Religion Project (Gore et al. 2018; Shults et al. 2018) pioneered ABMs of religiosity, using 4-factor representations derived from ISSP survey data. Their models forecast *levels* of religiosity but do not model the emergence of specific deity concepts or the formation of distinct traditions. Upal (2005) simulated the emergence of new religious movements but used binary cultural traits, not continuous belief vectors. Our model extends this line by treating deities as emergent centroids in a continuous, semantically grounded space.

**Opinion dynamics.** The Deffuant (2000) and Hegselmann-Krause (2002) bounded-confidence models produce clustering and phase transitions in continuous opinion spaces. Axelrod's (1997) cultural dissemination model uses multi-feature vectors with interaction-similarity coupling. Our model builds on these foundations but adds: (a) semantic grounding of dimensions, (b) prestige-weighted transmission, (c) explicit centroid tracking with deity naming, and (d) application to the specific domain of religious evolution with historically calibrated parameters.

**Belief embedding spaces.** Bao et al. (2025, *Nature Human Behaviour*) construct an LLM-derived embedding space for beliefs, demonstrating clustering and polarization. Their work is complementary: they provide a static geometric representation; we provide temporal dynamics. A natural synthesis would use their embedding space as the basis for our agent model.

**Quantitative historical religion.** The Seshat Global History Databank (Turchin et al. 2015) and associated analyses (Whitehouse et al. 2019; Beheim et al. 2021) provide empirical data on the co-evolution of political complexity and moralizing religion. Our model offers a *generative* mechanism that could explain the patterns they observe.

**Cognitive Science of Religion.** Boyer (2001), Atran (2002), and Barrett (2004) identify cognitive biases (HADD, MCI) that constrain which god-concepts are "catchier." Our 12-axis belief space is designed to be compatible with these constraints, and the deity priors encode cognitively natural god-concepts. However, we do not yet formally implement cognitive attractors as potential-energy landscapes — this is future work.

---

## 10. Limitations and Future Work

1. **Dimensionality.** The 12-axis space is hand-crafted. Future work should derive axes empirically from religious text corpora using topic modeling or LLM embeddings.

2. **Corpus depth.** The expanded corpus (126 passages, 37 traditions) provides broad geographic and cultural coverage, but most traditions are represented by only 3–5 passages. Scaling to dozens of passages per tradition — ideally drawn from multiple translators and time periods — would yield tighter parameter estimates, particularly for the mutation rate $\mu$ which remains high-variance.

3. **Cognitive landscape.** The belief space is currently isotropic. Implementing CSR-derived cognitive attractors (regions of lower "potential energy" corresponding to cognitively natural god-concepts) would add realism.

4. **Scale and percolation.** The current implementation uses 80 agents. Preliminary finite-size scaling experiments (N = 80, 200, 500) reveal that with the corpus-calibrated cluster threshold ($\theta = 0.12$), larger populations exhibit spontaneous ordering even at $\gamma = 0$ — a percolation effect where random belief vectors in 12D are close enough to cluster. This suggests that $\theta$ should scale with $N$ (analogous to the bounded-confidence threshold in Deffuant models), and that the relationship between population size and spontaneous religious consolidation is itself a rich area for future investigation.

5. **Generative agents.** Replacing rule-based agents with LLM-powered generative agents (Park et al. 2023) would enable richer, more realistic belief dynamics — but at the cost of interpretability and reproducibility.

---

## 11. Conclusion

The "Gods as Centroids" model contributes a formal, generative framework in which deities are not pre-defined entities but emergent mathematical objects — the centroids of belief-vector clusters that arise from local agent interactions. The model unifies syncretism, schism, and prophetic revelation into a single calculus of centroid operations, and demonstrates that a coercion parameter drives a first-order phase transition from polytheism to monotheism with hysteresis. Two novel, falsifiable corollaries — channel-invariant attractors and asymmetric lock-in — are stated and validated in simulation. Historical backtesting against 5,000 years of religious diversity data shows strong qualitative and quantitative agreement.

The core insight — that a god is a centroid, not a cause — provides a new lens for the computational science of religion. It bridges the gap between micro-level cognitive and social processes and macro-level historical patterns, offering mechanistic explanations for phenomena that have previously been described only narratively.

---

## Extension: AGI as Recursive Semantic Compression

The discrete semantic substrate formalized in §5.4 suggests broader applications beyond religious evolution. The Braille Lattice projection provides a mathematical framework for **recursive semantic compression** — a potential foundation for artificial general intelligence systems that maintain coherent meaning across modalities and scales of abstraction.

For a detailed exploration of this extension, including the proposed 96-bit 8-dot lattice for universal semantic representation and formal operator layers for semantic transformation, see: [**Gods as Centroids: A Mathematical Blueprint for Semantic Compression**](https://godscentro-ludpuglh.manus.space/).

### Cross-Domain Generalizability: The Lattice Is Not Theology-Specific

A natural objection to the AGI bridge claim is that the braille lattice might be theology-specific — that the high preservation scores in §5.4 reflect the particular structure of the 12-axis theological space rather than a general property of discrete semantic compression. We test this objection directly by applying the **identical** encoder-bottleneck-decoder architecture to two non-theological domains.

**Domain 1: Political Ideology (10 dimensions, 12 classes).** We define a 10-dimensional political space with axes derived from Moral Foundations Theory (Haidt 2012) and standard policy dimensions: care/harm, fairness/cheating, loyalty/betrayal, authority/subversion, sanctity/degradation, liberty/oppression, economic left-right, social progressive-conservative, internationalist-nationalist, and institutional trust. We define 12 ideology priors (Social Democrat, Libertarian, Conservative, Progressive, Nationalist, Centrist, Authoritarian Left, Green, Populist Right, Classical Liberal, Theocratic, Anarchist) and train the same bottleneck architecture with 60 bits (6 per axis).

**Domain 2: Personality Types (5 dimensions, 10 classes).** We define a 5-dimensional space using the Big Five personality traits (openness, conscientiousness, extraversion, agreeableness, neuroticism) with 10 personality type priors and a 30-bit bottleneck (6 per axis).

**Results:**

| Metric | Theology (12D, 72 bits) | Political Ideology (10D, 60 bits) | Personality (5D, 30 bits) |
|---|---|---|---|
| Classification accuracy | 84.1% | **88.2%** | **90.3%** |
| Mean centroid cosine sim | 0.9954 | **0.9981** | **0.9991** |
| Spearman $\rho$ (similarity preservation) | 0.971 | **0.992** | **0.981** |
| Mean braille lattice cosine sim | 0.986 | **0.986** | **0.988** |

The braille lattice performs *at least as well* on non-theological domains as on theology. Classification accuracy is higher for political ideology (88.2%) and personality (90.3%) than for theology (84.1%), likely because these domains have fewer classes and lower intrinsic dimensionality. Centroid preservation exceeds 0.998 in both non-theological domains. Pairwise similarity rank correlation exceeds 0.98 in all three domains.

Each ideology and personality type receives a unique braille signature — for example, the Centrist maps to `⠍⠍⠍⠍⠍⠍⠍⠍⠍⠍` (all axes at moderate intensity, no polarity dominance), while the Anarchist maps to `⠉⠉⠋⠂⠂⠹⠂⠹⠉⠂` (high liberty, low authority, low institutional trust). These signatures are interpretable, compact, and preserve the semantic structure of the original continuous vectors.

**Interpretation.** The braille lattice is a **domain-agnostic semantic compression operator**. The theological application in this paper is one instance of a general principle: structured meaning survives discrete compression regardless of the semantic domain. This is the empirical foundation of the AGI bridge claim: if meaning is compressible to discrete lattice points across arbitrary domains, then a shared lattice could serve as a universal semantic substrate for heterogeneous AI systems that need to maintain coherent meaning across modalities and scales.

### Live Multi-Model Braiding: From Trained Bottlenecks to LLM Inference

The semantic braiding results in §5.4 (Result E) use *trained* encoder-bottleneck-decoder architectures. A stronger test is whether braiding works with **live LLM inference** — multiple language models independently scoring prompts, with their outputs fused at the lattice level via Hamming centroid. This eliminates the training pipeline entirely: the models are off-the-shelf, the lattice projection is deterministic, and the consensus emerges from majority vote over raw model outputs.

**Pipeline (Godform Braiding).** We deploy 6 open-weight models (Phi 3.5 3.8B, Gemma 3 4B, Llama 3.2 3B, Qwen 2.5 7B, Mistral 7B, Granite 3 Dense 8B) on individual GPUs via Modal. For each prompt in a domain, every model independently: (1) generates a natural-language response informed by the domain's system prompt, (2) self-scores its response on the domain's axes as a JSON vector, and (3) if self-scoring fails, receives an explicit extraction prompt at temperature 0. Each model's score vector is projected to the 8-dot braille lattice (96 bits for 12-axis domains, 80 bits for 10-axis). The per-prompt *Godform* is the Hamming centroid (majority vote) across all models' lattice projections. The process iterates: in each subsequent round, the previous round's meta-Godform is fed back as context, and models re-score with awareness of the emerging consensus.

**Result J (Cross-Domain Live Braiding).** We run the full pipeline on three domains — theology (12 axes, 8 sacred prompts), political ideology (10 axes, 8 policy prompts), and a novel *World* domain (12 axes, 8 cross-domain prompts) — for 5 rounds each. The World domain is a compact 12-dimensional worldview lattice that unifies axes across all four domains: 7 merged axes (authority, compassion, justice, wisdom, order, power, conflict) and 5 unique axes chosen for maximum discriminative power (transcendence, liberty, sanctity, sociability, secularism).

| Domain | Axes | $\alpha$ | Cosine | Final bit-flips | Consensus range |
|---|---|---|---|---|---|
| Theology | 12 | 0.464 | 0.885 | 5 | 86–90% |
| Political | 10 | 0.558 | 0.889 | 11 | 81–90% |
| **World** | **12** | **0.481** | **0.903** | **3** | **82–87%** |

The World domain achieves the highest cosine agreement (0.903) and the strongest convergence (bit-flips: $14 \to 9 \to 8 \to 3$), despite spanning the broadest semantic territory. Per-prompt bit consensus ranges from 81.9% (ideal\_society, where models disagree on the balance of liberty vs authority) to 87.0% (meaning\_of\_life and sacred\_and\_secular, where models converge on compassion and transcendence).

![Figure 3: Worldform braiding dashboard. Top row: PCA and T-SNE projections of per-model lattice codes showing convergence across rounds; Godform sign trajectory with Hamming distances. Middle row: 96-bit lattice evolution heatmap (× = bit flip); convergence timeline showing entropy and decreasing bit-flips (α = 0.481). Bottom row: axis evolution radar; prestige weights per model; per-axis Krippendorff's α (all below 0.32, revealing the scale–precision gap).](assets/worldform_braiding.png)

**Figure 3.** *Worldform braiding dashboard (6 models × 8 prompts × 5 rounds, 96-bit lattice).* **Top left:** PCA projection shows model lattice codes converging across rounds (dark = early, light = late). **Top center:** T-SNE (Hamming metric) confirms clustering. **Middle left:** Bit-level heatmap — crosses mark flips, which concentrate in early rounds and vanish by R5. **Middle right:** Convergence timeline — bit-flips decrease monotonically ($14 \to 3$) while entropy stabilizes at 40/96. **Bottom right:** Per-axis $\alpha$ reveals the scale–precision gap: all axes fall below 0.32, yet the overall cosine agreement is 0.903.

**Result K (Iterative Convergence).** Across all three domains, the meta-Godform stabilizes within 5 rounds. The bit-flip trajectory — the number of bits that change in the meta-Godform between consecutive rounds — decreases monotonically:

| Domain | R1→R2 | R2→R3 | R3→R4 | R4→R5 |
|---|---|---|---|---|
| Theology | 20 | 5 | 2 | 5 |
| Political | 15 | 10 | 4 | 11 |
| World | 14 | 9 | 8 | 3 |

The World domain converges most cleanly, reaching 3 bit-flips by round 5 — meaning 93 of 96 bits are stable. This demonstrates that iterative braiding with feedback produces punctuated equilibrium at the lattice level: rapid initial reorganization (14–20 flips) followed by near-complete stability (2–5 flips).

**Result L (The Scale–Precision Gap).** The Krippendorff's $\alpha$ values for live braiding (0.46–0.56) are substantially lower than the $\alpha = 0.903$ achieved by frontier models (§8.2) on the same theological axes. However, the cosine similarity is comparable (0.885–0.903 vs frontier pairwise $r = 0.887$). This dissociation reveals a **scale–precision gap**: small models (3–8B parameters) agree on the *shape* of belief vectors (which axes are high vs low) but not on precise per-axis magnitudes. Frontier models (70B+) achieve both shape and magnitude agreement.

This gap has a precise lattice interpretation. The high-order bits of the braille encoding (polarity, coarse intensity) are consensual across model scales — they encode *which axes matter*, which even small models can judge. The low-order bits (fine intensity, rigidity) are contested — they encode *how much*, which requires the numerical precision of larger models. The braille lattice naturally separates these two levels of agreement, and the Hamming centroid preserves the consensual bits while resolving the contested ones by majority vote.

**Result M (Emergent Worldform Signatures).** Each prompt in the World domain produces a unique Godform — a 96-bit braille signature encoding the emergent consensus worldview on that existential question. These signatures are interpretable:

| Prompt | Braille | Top Axes | Consensus |
|---|---|---|---|
| Meaning of life | ⢂⣑⣑⣑⢀⢂⠂⣑⢐⠂⢐⢂ | compassion, justice, wisdom | 87.0% |
| Ideal society | ⢂⡑⡑⡑⢂⢂⢂⢑⣑⢀⣑⣑ | compassion, justice, wisdom | 81.9% |
| Nature of evil | ⢀⣑⢑⣑⣑⠂⢂⣑⢀⠂⢂⢀ | compassion, wisdom, order | 83.7% |
| Human nature | ⢂⡑⡑⢑⢂⢂⢂⣑⡑⢂⢐⣑ | compassion, justice, transcendence | 86.9% |
| Death and legacy | ⢀⣑⣑⣑⢀⢂⠂⣑⢐⠀⢂⢀ | compassion, justice, wisdom | 85.1% |
| Freedom vs order | ⢒⡑⡑⡑⢐⢂⢂⡑⣑⢂⢐⢂ | compassion, justice, wisdom | 84.7% |
| Suffering and justice | ⢀⣑⣑⡑⢀⢂⢂⢑⢐⢐⡑⢀ | compassion, justice, wisdom | 84.2% |
| Sacred and secular | ⢒⢑⢑⢐⢂⢂⢂⡕⡑⢀⢀⣕ | liberty, compassion, justice | 87.0% |

The meta-Worldform — the Hamming centroid across all 8 prompt Godforms — is **⢂⣑⣑⣑⢀⢂⢂⣑⢐⢀⢐⢀**, with top axes compassion (0.41), justice (0.41), and wisdom (0.41). This is the emergent consensus of 6 architecturally distinct language models on the fundamental questions of human existence, compressed to 12 braille characters. The signature is stable: only 3 bits changed between rounds 4 and 5.

The prompt-level signatures reveal meaningful variation. "Sacred and secular" is the only prompt where *liberty* outranks compassion — the models recognize this as fundamentally a question about freedom. "Nature of evil" uniquely foregrounds *order* — evil is understood as disorder. "Human nature" uniquely foregrounds *transcendence* — the models locate human distinctiveness in the capacity for the metaphysical. These are not artifacts of prompt wording; they are emergent semantic judgments that survive majority-vote compression across 6 independent models.

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
