# Gods as Centroids — Honest Novelty Assessment

## The Prior Art (What Already Exists)

### 1. Agent-Based Models of Religion (Shults, Wildman, Gore et al.)
The **Modeling Religion Project** (Center for Mind and Culture) has been running since ~2016.
Their flagship paper (Gore et al. 2018, JASSS) builds an ABM that:
- Represents agents with **4 religiosity factors** (formation, practice, supernatural beliefs, belief in God) derived from ISSP survey data
- Uses social networks for influence propagation
- Models existential security as an environmental driver
- **Forecasts** changes in religiosity for real countries
- Uses **multi-dimensional** representation of belief (not scalar)

**What they DON'T do:** They don't treat deities as emergent entities. Their model tracks *religiosity levels*, not the formation of specific god-concepts. No clustering, no centroids, no deity emergence.

### 2. Belief Embedding Spaces (Nature, 2025)
A recent Nature Human Behaviour paper constructs a **semantic embedding space for beliefs** using LLMs fine-tuned on debate.org data. They:
- Represent beliefs as vectors in high-dimensional space
- Show clustering and polarization patterns
- Predict individual stances from belief embeddings
- Use cosine similarity for belief proximity

**What they DON'T do:** No agent-based dynamics, no temporal evolution, no religious content specifically, no deity emergence, no phase transitions.

### 3. Axelrod's Cultural Dissemination Model (1997)
The classic. Agents have multi-feature cultural vectors. Interaction probability ∝ similarity. Produces cultural clustering and phase transitions between monoculture and multiculture.

**What they DON'T do:** No semantic grounding of dimensions, no deity/centroid emergence, no religious-specific dynamics (syncretism, schism, prophecy), no cognitive science grounding.

### 4. Opinion Dynamics / Bounded Confidence Models
Deffuant, Hegselmann-Krause, etc. Continuous opinion vectors, bounded confidence thresholds, clustering dynamics, phase transitions.

**What they DON'T do:** No religious application, no deity emergence, no semantic axes, no prestige/charisma dynamics.

### 5. Seshat / Turchin / Whitehouse (Quantitative Historical Religion)
The Seshat Global History Databank + cliodynamics approach. Quantitative data on religious complexity across history. The (retracted) "moralizing gods" Nature paper attempted to show complex societies precede moralizing gods.

**What they DON'T do:** No generative simulation, no agent-based model, no vector space representation. Purely empirical/statistical.

### 6. Cognitive Science of Religion (Boyer, Atran, Barrett)
HADD, MCI concepts, cognitive attractors. Theoretical frameworks for why certain god-concepts are "catchier" than others.

**What they DON'T do:** No computational implementation, no simulation, no formal model.

---

## What IS Genuinely Novel in "Gods as Centroids"

After surveying the field, here's what's actually new:

### ✅ Novel Contribution 1: Deities as Emergent Centroids
**No prior work treats deities themselves as emergent mathematical objects.** Existing ABMs model *religiosity levels* (how religious someone is) or *cultural traits* (which features they have). Nobody has formalized the idea that a "god" is the centroid of a belief cluster — an entity that exists only as a statistical summary of its followers' beliefs, and that co-evolves with them. This is the core insight and it IS genuinely new.

### ✅ Novel Contribution 2: Semantic Grounding of the Belief Space
The 12-axis theological vector space (authority, transcendence, care, justice, wisdom, power, fertility, war, death, creation, nature, order) is grounded in both comparative religion and CSR. Prior opinion dynamics models use abstract dimensions. The Nature belief-embedding paper uses LLM-derived dimensions. Nobody has hand-crafted a *theologically interpretable* vector space specifically designed to capture the axes along which god-concepts vary.

### ✅ Novel Contribution 3: Unified Dynamics of Syncretism, Schism, and Prophecy
Prior models handle cultural convergence OR divergence. Nobody has unified fusion (syncretism), fission (schism), and punctuated mutation (prophecy) into a single formal framework with explicit centroid operations. The merge/split/perturb calculus on cluster centroids is a clean formal contribution.

### ✅ Novel Contribution 4: Coercion-Driven Phase Transition with Hysteresis
While phase transitions exist in opinion dynamics, nobody has applied them specifically to the polytheism→monotheism transition with an explicit coercion parameter, or identified the *hysteresis* (asymmetric lock-in) as a formal prediction. The prediction that reducing coercion doesn't restore polytheism is testable and novel.

### ✅ Novel Contribution 5: Accessibility Corollary (Channel-Invariant Attractors)
The prediction that sensory-restricted agents converge to the same centroids is genuinely novel and has implications for disability studies and religious accessibility. No prior work addresses this.

### ⚠️ Weak/Non-Novel Claims
- "Swarm intelligence applied to religion" — too vague, Axelrod already did cultural swarms
- "Vector space models of belief" — the Nature 2025 paper does this better with real data
- "Agent-based modeling of religion" — Shults/Wildman have been doing this for a decade
- The CSR connection (HADD, MCI) — mentioned but not formally implemented
- The GABM/LLM future work section — speculative, not a contribution

---

## The Honest Verdict

**The paper as written is ~60% literature review and ~40% novel framework.** The novelty IS real but it's buried under too much textbook explanation of K-means, swarm intelligence, and vector spaces. A reviewer would say: "This reads like a survey paper with an interesting idea sketched at the end."

### What Needs to Change for Publication

1. **Cut the tutorials.** A reviewer at JASSS or JCSS doesn't need K-means explained. Cut Sections I-II by 70%.

2. **Lead with the formal model.** Define the mathematical framework precisely: the belief space B, the agent state, the centroid operator, the merge/split/mutate operations, the coercion landscape warping. Make it look like a math paper, not a survey.

3. **Engage prior art honestly.** Cite Shults/Wildman, the Nature belief-embedding paper, Axelrod, and Deffuant explicitly. Show exactly where your model differs.

4. **Add empirical validation.** The simulation exists. Run it. Show that the phase transition actually occurs. Show the hysteresis. Show the Accessibility Corollary holds. Plot the results. Without this, it's a thought experiment.

5. **State falsifiable predictions.** The hysteresis prediction is testable against Seshat data. The accessibility corollary is testable. State them as formal hypotheses.

6. **Target the right venue.** This isn't a CS paper (no algorithmic contribution). It's a computational social science paper. Target: JASSS, Journal of Cognition and Culture, Religion Brain & Behavior, or JCSS.
