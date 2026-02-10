use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// Semantic axes for deity representation
const AXES: &[&str] = &[
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
];

// Historical deity semantic priors
fn get_deity_priors() -> HashMap<String, Vec<f64>> {
    let mut priors = HashMap::new();
    
    // Major pantheon representatives with normalized semantic vectors
    priors.insert("zeus".to_string(), vec![0.9, 0.8, 0.3, 0.7, 0.6, 0.9, 0.2, 0.8, 0.1, 0.4, 0.3, 0.8]);
    priors.insert("odin".to_string(), vec![0.8, 0.7, 0.4, 0.6, 0.9, 0.7, 0.1, 0.9, 0.8, 0.3, 0.2, 0.5]);
    priors.insert("amun".to_string(), vec![0.9, 0.9, 0.6, 0.8, 0.8, 0.8, 0.3, 0.2, 0.1, 0.9, 0.1, 0.9]);
    priors.insert("marduk".to_string(), vec![0.9, 0.6, 0.5, 0.9, 0.7, 0.9, 0.1, 0.8, 0.3, 0.7, 0.1, 0.9]);
    priors.insert("indra".to_string(), vec![0.8, 0.5, 0.4, 0.7, 0.6, 0.9, 0.2, 0.9, 0.2, 0.3, 0.4, 0.6]);
    priors.insert("shango".to_string(), vec![0.7, 0.4, 0.3, 0.8, 0.5, 0.8, 0.1, 0.7, 0.2, 0.2, 0.6, 0.5]);
    priors.insert("kami".to_string(), vec![0.3, 0.8, 0.8, 0.4, 0.7, 0.4, 0.6, 0.1, 0.1, 0.5, 0.9, 0.8]);
    priors.insert("manitou".to_string(), vec![0.2, 0.9, 0.9, 0.3, 0.8, 0.3, 0.7, 0.1, 0.2, 0.6, 0.9, 0.4]);
    
    // Normalize all vectors
    for vector in priors.values_mut() {
        let norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }
    
    priors
}

fn get_theonyms() -> Vec<String> {
    vec![
        "zeus", "odin", "amun", "marduk", "indra", "shango", "kami", "manitou",
        "apollo", "freya", "ptah", "ishtar", "perun", "teotl", "nut", "hades",
        "thor", "isis", "ra", "quetzal", "tyr", "bast", "lugh", "brigid",
        "taranis", "nana", "enar", "yah", "baal"
    ].into_iter().map(String::from).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub n: usize,
    pub steps_per_generation: usize,
    pub max_message_len: usize,
    pub learning_rate: f64,
    pub penalty_rate: f64,
    pub prestige_alpha: f64,
    pub ritual_period: usize,
    pub ritual_bonus: f64,
    pub base_success_thresh: f64,
    pub mutation_rate: f64,
    pub exploration_eps: f64,
    pub generation_mix_k: usize,
    pub seed: u64,
    pub topo_window: usize,
    
    // Deity priors and social dynamics
    pub use_deity_priors: bool,
    pub belief_influence: f64,
    pub coercion: f64,
    pub social_network: String,
    pub social_k: usize,
    pub social_p: f64,
    pub cluster_update_freq: usize,
    pub cluster_threshold: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n: 50,
            steps_per_generation: 2500,
            max_message_len: 3,
            learning_rate: 0.08,
            penalty_rate: 0.02,
            prestige_alpha: 0.20,
            ritual_period: 50,
            ritual_bonus: 0.10,
            base_success_thresh: 0.58,
            mutation_rate: 0.08,
            exploration_eps: 0.10,
            generation_mix_k: 3,
            seed: 42,
            topo_window: 200,
            use_deity_priors: true,
            belief_influence: 0.15,
            coercion: 0.0,
            social_network: "small_world".to_string(),
            social_k: 4,
            social_p: 0.1,
            cluster_update_freq: 100,
            cluster_threshold: 0.4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Context {
    pub task: String,
    pub role: String,
    pub place: String,
    pub tod: String,
    pub vec: Vec<f64>,
}

impl Context {
    fn new(rng: &mut StdRng) -> Self {
        let tasks = vec!["hunt", "gather", "build", "trade", "ritual", "teach"];
        let roles = vec!["leader", "shaman", "warrior", "elder", "child", "healer"];
        let places = vec!["forest", "river", "mountain", "cave", "village", "field"];
        let tods = vec!["dawn", "morning", "noon", "evening", "dusk", "night"];
        
        let task = tasks.choose(rng).unwrap().to_string();
        let role = roles.choose(rng).unwrap().to_string();
        let place = places.choose(rng).unwrap().to_string();
        let tod = tods.choose(rng).unwrap().to_string();
        
        // Generate semantic vector
        let mut vec = vec![0.0; AXES.len()];
        for v in vec.iter_mut() {
            *v = rng.gen::<f64>() * 2.0 - 1.0;
        }
        normalize_vector(&mut vec);
        
        Context { task, role, place, tod, vec }
    }
}

#[derive(Debug, Clone)]
pub struct Agent {
    pub id: usize,
    pub belief: Vec<f64>,
    pub w: f64,
    pub assoc: HashMap<String, Vec<f64>>,
    pub freq: HashMap<String, usize>,
}

impl Agent {
    fn new(id: usize, belief: Vec<f64>) -> Self {
        Agent {
            id,
            belief,
            w: 1.0,
            assoc: HashMap::new(),
            freq: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Metrics {
    pub zipf_slope: f64,
    pub heaps_k: f64,
    pub cond_entropy: f64,
    pub topo_similarity: f64,
    pub churn: f64,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            zipf_slope: 0.0,
            heaps_k: 0.0,
            cond_entropy: 0.0,
            topo_similarity: 0.0,
            churn: 0.0,
        }
    }
}

pub struct SwarmKernel {
    cfg: Config,
    rng: StdRng,
    agents: Vec<Agent>,
    t: usize,
    gen: usize,
    tokens: Vec<String>,
    types: HashMap<String, usize>,
    bigrams: HashMap<(String, String), usize>,
    last_token: Option<String>,
    ctx_window: VecDeque<(Vec<f64>, Vec<f64>)>,
    pref_form: HashMap<usize, Option<String>>,
    metrics: Metrics,
    social_graph: HashMap<usize, Vec<usize>>,
    clusters: Vec<Vec<usize>>,
    centroids: Vec<Vec<f64>>,
}

impl SwarmKernel {
    pub fn new(cfg: Config) -> Self {
        let rng = StdRng::seed_from_u64(cfg.seed);
        let mut kernel = SwarmKernel {
            cfg: cfg.clone(),
            rng,
            agents: Vec::new(),
            t: 0,
            gen: 0,
            tokens: Vec::new(),
            types: HashMap::new(),
            bigrams: HashMap::new(),
            last_token: None,
            ctx_window: VecDeque::with_capacity(cfg.topo_window),
            pref_form: HashMap::new(),
            metrics: Metrics::default(),
            social_graph: HashMap::new(),
            clusters: Vec::new(),
            centroids: Vec::new(),
        };
        
        kernel.init_agents();
        kernel.build_social_network();
        kernel
    }
    
    fn init_agents(&mut self) {
        let deity_priors = get_deity_priors();
        let theonyms = get_theonyms();
        
        for i in 0..self.cfg.n {
            // Initialize belief from deity mixture if using priors
            let belief = if self.cfg.use_deity_priors {
                let chosen_count = self.rng.gen_range(1..=2);
                let chosen: Vec<_> = deity_priors.keys()
                    .choose_multiple(&mut self.rng, chosen_count)
                    .into_iter().collect();
                
                let mut belief = vec![0.0; AXES.len()];
                for deity_name in chosen {
                    if let Some(deity_vec) = deity_priors.get(deity_name) {
                        for (j, &val) in deity_vec.iter().enumerate() {
                            belief[j] += val;
                        }
                    }
                }
                
                // Add jitter
                for b in belief.iter_mut() {
                    *b += self.rng.gen::<f64>() * 0.2 - 0.1;
                }
                normalize_vector(&mut belief);
                belief
            } else {
                random_unit_vector(&mut self.rng, AXES.len())
            };
            
            let mut agent = Agent::new(i, belief);
            
            // Seed with theonyms
            for name in &theonyms {
                let vec = if self.cfg.use_deity_priors && deity_priors.contains_key(name) {
                    jitter_vector(deity_priors.get(name).unwrap(), &mut self.rng, 0.1)
                } else {
                    random_unit_vector(&mut self.rng, AXES.len())
                };
                agent.assoc.insert(name.clone(), vec);
            }
            
            self.agents.push(agent);
            self.pref_form.insert(i, None);
        }
    }
    
    fn build_social_network(&mut self) {
        // Watts-Strogatz small-world network
        for i in 0..self.cfg.n {
            self.social_graph.insert(i, Vec::new());
        }
        
        let k = self.cfg.social_k / 2;
        
        // Create ring lattice
        for i in 0..self.cfg.n {
            for j in 1..=k {
                let neighbor = (i + j) % self.cfg.n;
                self.social_graph.get_mut(&i).unwrap().push(neighbor);
                self.social_graph.get_mut(&neighbor).unwrap().push(i);
            }
        }
        
        // Rewire with probability p
        let mut to_rewire = Vec::new();
        for i in 0..self.cfg.n {
            let neighbors = self.social_graph.get(&i).unwrap().clone();
            for &neighbor in &neighbors {
                if self.rng.gen::<f64>() < self.cfg.social_p {
                    to_rewire.push((i, neighbor));
                }
            }
        }
        
        for (i, old_neighbor) in to_rewire {
            // Remove old edge
            self.social_graph.get_mut(&i).unwrap().retain(|&x| x != old_neighbor);
            self.social_graph.get_mut(&old_neighbor).unwrap().retain(|&x| x != i);
            
            // Add new edge
            let candidates: Vec<usize> = (0..self.cfg.n)
                .filter(|&x| x != i && !self.social_graph.get(&i).unwrap().contains(&x))
                .collect();
            
            if !candidates.is_empty() {
                let new_neighbor = *candidates.choose(&mut self.rng).unwrap();
                self.social_graph.get_mut(&i).unwrap().push(new_neighbor);
                self.social_graph.get_mut(&new_neighbor).unwrap().push(i);
            }
        }
    }
    
    pub fn run(&mut self, steps: usize) {
        for step in 0..steps {
            self.step();
            
            if step % 500 == 0 {
                self.update_metrics();
                println!("t={:6} gen={} | zipf={:+.2} heaps={:.3} entropy={:.2} topo={:+.2} churn={:.2}",
                    self.t, self.gen, self.metrics.zipf_slope, self.metrics.heaps_k,
                    self.metrics.cond_entropy, self.metrics.topo_similarity, self.metrics.churn);
            }
            
            if self.t % self.cfg.cluster_update_freq == 0 {
                self.update_clusters();
            }
        }
    }
    
    fn step(&mut self) {
        self.t += 1;
        
        // Select speaker weighted by prestige
        let weights: Vec<f64> = self.agents.iter().map(|a| a.w).collect();
        let speaker_idx = weighted_choice(&mut self.rng, &weights);
        let speaker = &self.agents[speaker_idx];
        
        // Select hearers from social network
        let hearers = self.select_hearers(speaker_idx);
        
        // Generate context
        let ctx = Context::new(&mut self.rng);
        
        // Produce message
        let msg = self.produce(speaker_idx, &ctx);
        
        // Update token statistics
        for token in &msg {
            self.tokens.push(token.clone());
            *self.types.entry(token.clone()).or_insert(0) += 1;
            
            if let Some(ref last) = self.last_token {
                *self.bigrams.entry((last.clone(), token.clone())).or_insert(0) += 1;
            }
            self.last_token = Some(token.clone());
        }
        
        // Interaction and learning
        let success = self.interact(speaker_idx, &hearers, &ctx, &msg);
        self.learn_from(speaker_idx, &msg, &ctx, success);
        for &hearer_idx in &hearers {
            self.learn_from(hearer_idx, &msg, &ctx, success);
        }
        
        // Update prestige
        self.update_prestige(&[speaker_idx], success);
        self.update_prestige(&hearers, success);
        
        // Mutations
        if self.rng.gen::<f64>() < self.cfg.mutation_rate {
            let agent_idx = self.rng.gen_range(0..self.agents.len());
            self.mutate_agent(agent_idx);
        }
    }
    
    fn select_hearers(&mut self, speaker_idx: usize) -> Vec<usize> {
        let neighbors = self.social_graph.get(&speaker_idx).cloned().unwrap_or_default();
        
        if neighbors.is_empty() {
            return (0..self.cfg.n)
                .filter(|&i| i != speaker_idx)
                .choose_multiple(&mut self.rng, 2.min(self.cfg.n - 1));
        }
        
        if self.cfg.coercion > 0.0 {
            // Weight by belief similarity
            let speaker_belief = &self.agents[speaker_idx].belief;
            let mut weights = Vec::new();
            
            for &neighbor_idx in &neighbors {
                let neighbor_belief = &self.agents[neighbor_idx].belief;
                let sim = cosine_similarity(speaker_belief, neighbor_belief);
                let weight = (sim * (1.0 + 9.0 * self.cfg.coercion)).exp();
                weights.push(weight);
            }
            
            let mut chosen_indices = Vec::new();
            let total_weight: f64 = weights.iter().sum();
            for _ in 0..2.min(neighbors.len()) {
                let mut r = self.rng.gen::<f64>() * total_weight;
                for (i, &weight) in weights.iter().enumerate() {
                    r -= weight;
                    if r <= 0.0 {
                        chosen_indices.push(i);
                        break;
                    }
                }
            }
            
            chosen_indices.into_iter().map(|i| neighbors[i]).collect()
        } else {
            let len = neighbors.len();
            neighbors.into_iter()
                .choose_multiple(&mut self.rng, 2.min(len))
        }
    }
    
    fn produce(&mut self, agent_idx: usize, ctx: &Context) -> Vec<String> {
        let agent = &self.agents[agent_idx];
        let mut scored = Vec::new();
        let total_freq: usize = agent.freq.values().sum();
        
        for (form, vec) in &agent.assoc {
            let ctx_score = dot_product(vec, &ctx.vec);
            let belief_score = dot_product(vec, &agent.belief) * self.cfg.belief_influence;
            let freq_score = 0.1 * ((agent.freq.get(form).unwrap_or(&0) + 1) as f64 / (total_freq + 1) as f64).ln();
            let score = ctx_score + belief_score + freq_score;
            scored.push((form.clone(), score));
        }
        
        let mut msg = Vec::new();
        for _ in 0..self.cfg.max_message_len {
            let choice = self.softmax_choice(&scored);
            msg.push(choice);
        }
        msg
    }
    
    fn softmax_choice(&mut self, items: &[(String, f64)]) -> String {
        if self.rng.gen::<f64>() < self.cfg.exploration_eps {
            return items.choose(&mut self.rng).unwrap().0.clone();
        }
        
        let max_score = items.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = items.iter()
            .map(|(_, s)| (s - max_score).exp())
            .collect();
        let sum: f64 = exp_scores.iter().sum();
        
        let r = self.rng.gen::<f64>() * sum;
        let mut acc = 0.0;
        for (i, &exp_score) in exp_scores.iter().enumerate() {
            acc += exp_score;
            if acc >= r {
                return items[i].0.clone();
            }
        }
        items.last().unwrap().0.clone()
    }
    
    fn interact(&mut self, _speaker_idx: usize, _hearers: &[usize], _ctx: &Context, _msg: &[String]) -> bool {
        // Simplified success model
        self.rng.gen::<f64>() > self.cfg.base_success_thresh
    }
    
    fn learn_from(&mut self, agent_idx: usize, msg: &[String], ctx: &Context, success: bool) {
        let lr = if success { self.cfg.learning_rate } else { -self.cfg.penalty_rate };
        
        for token in msg {
            if !self.agents[agent_idx].assoc.contains_key(token) {
                self.agents[agent_idx].assoc.insert(token.clone(), random_unit_vector(&mut self.rng, AXES.len()));
            }
            
            let assoc_vec = self.agents[agent_idx].assoc.get_mut(token).unwrap();
            for (i, &ctx_val) in ctx.vec.iter().enumerate() {
                assoc_vec[i] += lr * ctx_val;
            }
            normalize_vector(assoc_vec);
            
            *self.agents[agent_idx].freq.entry(token.clone()).or_insert(0) += 1;
        }
    }
    
    fn update_prestige(&mut self, agent_indices: &[usize], success: bool) {
        let delta = if success { self.cfg.prestige_alpha } else { -self.cfg.prestige_alpha * 0.3 };
        for &idx in agent_indices {
            self.agents[idx].w = (self.agents[idx].w * (1.0 + delta)).clamp(0.1, 10.0);
        }
    }
    
    fn mutate_agent(&mut self, _agent_idx: usize) {
        // Simplified mutation - could add morphological operations
    }
    
    fn update_clusters(&mut self) {
        self.centroids.clear();
        self.clusters.clear();
        
        for agent in &self.agents {
            if self.centroids.is_empty() {
                self.centroids.push(agent.belief.clone());
                self.clusters.push(vec![agent.id]);
                continue;
            }
            
            let distances: Vec<f64> = self.centroids.iter()
                .map(|c| 1.0 - cosine_similarity(&agent.belief, c))
                .collect();
            
            let (best_idx, &min_dist) = distances.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            
            if min_dist < self.cfg.cluster_threshold {
                self.clusters[best_idx].push(agent.id);
            } else {
                self.centroids.push(agent.belief.clone());
                self.clusters.push(vec![agent.id]);
            }
        }
        
        // Recalculate centroids
        let mut new_centroids = Vec::new();
        for cluster in &self.clusters {
            if cluster.is_empty() {
                continue;
            }
            
            let mut centroid = vec![0.0; AXES.len()];
            for &agent_id in cluster {
                for (i, &val) in self.agents[agent_id].belief.iter().enumerate() {
                    centroid[i] += val;
                }
            }
            
            for c in centroid.iter_mut() {
                *c /= cluster.len() as f64;
            }
            normalize_vector(&mut centroid);
            new_centroids.push(centroid);
        }
        self.centroids = new_centroids;
    }
    
    fn update_metrics(&mut self) {
        // Simplified metrics calculation
        self.metrics.zipf_slope = self.calculate_zipf_slope();
        self.metrics.heaps_k = self.calculate_heaps_k();
        self.metrics.cond_entropy = self.calculate_conditional_entropy();
        self.metrics.topo_similarity = 0.0; // Placeholder
        self.metrics.churn = 0.0; // Placeholder
    }
    
    fn calculate_zipf_slope(&self) -> f64 {
        if self.types.len() < 2 {
            return 0.0;
        }
        
        let mut counts: Vec<usize> = self.types.values().cloned().collect();
        counts.sort_by(|a, b| b.cmp(a));
        
        let mut log_ranks = Vec::new();
        let mut log_freqs = Vec::new();
        
        for (i, &count) in counts.iter().enumerate() {
            if count > 0 {
                log_ranks.push(((i + 1) as f64).ln());
                log_freqs.push((count as f64).ln());
            }
        }
        
        if log_ranks.len() < 2 {
            return 0.0;
        }
        
        // Simple linear regression
        let n = log_ranks.len() as f64;
        let sum_x: f64 = log_ranks.iter().sum();
        let sum_y: f64 = log_freqs.iter().sum();
        let sum_xy: f64 = log_ranks.iter().zip(&log_freqs).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = log_ranks.iter().map(|x| x * x).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        slope
    }
    
    fn calculate_heaps_k(&self) -> f64 {
        if self.tokens.is_empty() {
            return 0.0;
        }
        
        let types = self.types.len() as f64;
        let tokens = self.tokens.len() as f64;
        
        if tokens <= 1.0 {
            return 0.0;
        }
        
        types / tokens.powf(0.5) // Simplified Heaps law
    }
    
    fn calculate_conditional_entropy(&self) -> f64 {
        if self.bigrams.is_empty() {
            return 0.0;
        }
        
        let mut prev_counts = HashMap::new();
        for ((prev, _), count) in &self.bigrams {
            *prev_counts.entry(prev.clone()).or_insert(0) += count;
        }
        
        let mut entropy = 0.0;
        let total_bigrams: usize = self.bigrams.values().sum();
        
        for (prev, prev_count) in prev_counts {
            let p_prev = prev_count as f64 / total_bigrams as f64;
            let mut conditional_entropy = 0.0;
            
            for ((p, _next), count) in &self.bigrams {
                if p == &prev {
                    let p_next_given_prev = *count as f64 / prev_count as f64;
                    if p_next_given_prev > 0.0 {
                        conditional_entropy -= p_next_given_prev * p_next_given_prev.log2();
                    }
                }
            }
            
            entropy += p_prev * conditional_entropy;
        }
        
        entropy
    }
}

// Utility functions
fn normalize_vector(vec: &mut [f64]) {
    let norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

fn random_unit_vector(rng: &mut StdRng, dim: usize) -> Vec<f64> {
    let mut vec: Vec<f64> = (0..dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
    normalize_vector(&mut vec);
    vec
}

fn jitter_vector(base: &[f64], rng: &mut StdRng, jitter: f64) -> Vec<f64> {
    let mut vec: Vec<f64> = base.iter()
        .map(|&x| x + (rng.gen::<f64>() * 2.0 - 1.0) * jitter)
        .collect();
    normalize_vector(&mut vec);
    vec
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot = dot_product(a, b);
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn weighted_choice(rng: &mut StdRng, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    let mut r = rng.gen::<f64>() * total;
    
    for (i, &weight) in weights.iter().enumerate() {
        r -= weight;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}
