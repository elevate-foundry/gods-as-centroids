#include "kernel.h"
#include "agent.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iomanip>

const std::vector<std::string> AXES = {
    "authority", "transcendence", "care", "justice", "wisdom", "power",
    "fertility", "war", "death", "creation", "nature", "order"
};

// JSON serialization for Config - simplified stub
void to_json(json& j, const Config& c) {
    // Simplified for compilation
}

void from_json(const json& j, Config& c) {
    // Simplified for compilation - using defaults
}

Context::Context(std::mt19937& rng) {
    std::vector<std::string> tasks = {"hunt", "gather", "build", "trade", "war", "ritual"};
    std::vector<std::string> roles = {"leader", "warrior", "shaman", "elder", "child", "stranger"};
    std::vector<std::string> places = {"forest", "river", "mountain", "cave", "village", "temple"};
    std::vector<std::string> tods = {"dawn", "morning", "noon", "evening", "night", "midnight"};
    
    std::uniform_int_distribution<int> dist;
    
    dist = std::uniform_int_distribution<int>(0, tasks.size() - 1);
    task = tasks[dist(rng)];
    
    dist = std::uniform_int_distribution<int>(0, roles.size() - 1);
    role = roles[dist(rng)];
    
    dist = std::uniform_int_distribution<int>(0, places.size() - 1);
    place = places[dist(rng)];
    
    dist = std::uniform_int_distribution<int>(0, tods.size() - 1);
    tod = tods[dist(rng)];
    
    vec = random_unit_vector(rng, AXES.size());
}

SwarmKernel::SwarmKernel(const Config& cfg) : cfg_(cfg), rng_(cfg.seed) {
    init_agents();
    build_social_network();
}

void SwarmKernel::init_agents() {
    auto deity_priors = get_deity_priors();
    auto theonyms = get_theonyms();
    
    agents_.reserve(cfg_.n);
    for (int i = 0; i < cfg_.n; ++i) {
        std::vector<double> belief;
        
        if (cfg_.use_deity_priors && !deity_priors.empty()) {
            std::uniform_int_distribution<int> dist(0, theonyms.size() - 1);
            std::string deity = theonyms[dist(rng_)];
            
            if (deity_priors.find(deity) != deity_priors.end()) {
                belief = jitter_vector(deity_priors[deity], rng_, 0.1);
            } else {
                belief = random_unit_vector(rng_, AXES.size());
            }
        } else {
            belief = random_unit_vector(rng_, AXES.size());
        }
        
        agents_.emplace_back(i, std::move(belief));
    }
}

void SwarmKernel::build_social_network() {
    social_graph_.clear();
    
    if (cfg_.social_network == "small_world") {
        // Initialize ring lattice
        for (int i = 0; i < cfg_.n; ++i) {
            social_graph_[i] = std::vector<int>();
            for (int j = 1; j <= cfg_.social_k / 2; ++j) {
                int neighbor1 = (i + j) % cfg_.n;
                int neighbor2 = (i - j + cfg_.n) % cfg_.n;
                social_graph_[i].push_back(neighbor1);
                social_graph_[i].push_back(neighbor2);
            }
        }
        
        // Rewire edges with probability p
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<int> node_dist(0, cfg_.n - 1);
        
        for (int i = 0; i < cfg_.n; ++i) {
            auto& neighbors = social_graph_[i];
            for (size_t j = 0; j < neighbors.size(); ++j) {
                if (prob_dist(rng_) < cfg_.social_p) {
                    int old_neighbor = neighbors[j];
                    int new_neighbor;
                    do {
                        new_neighbor = node_dist(rng_);
                    } while (new_neighbor == i || has_edge(i, new_neighbor));
                    
                    remove_edge(i, old_neighbor);
                    neighbors[j] = new_neighbor;
                }
            }
        }
    } else {
        // Complete graph fallback
        for (int i = 0; i < cfg_.n; ++i) {
            social_graph_[i] = std::vector<int>();
            for (int j = 0; j < cfg_.n; ++j) {
                if (i != j) {
                    social_graph_[i].push_back(j);
                }
            }
        }
    }
}

void SwarmKernel::run(int steps) {
    for (int i = 0; i < steps; ++i) {
        step();
        
        if (t_ % 500 == 1 || t_ == steps) {
            update_metrics();
            std::cout << "t=" << std::setw(6) << t_ << " gen=" << gen_ 
                      << " | zipf=" << std::fixed << std::setprecision(2) << std::showpos << metrics_.zipf_slope
                      << " heaps=" << std::noshowpos << metrics_.heaps_k
                      << " entropy=" << metrics_.cond_entropy
                      << " topo=" << std::showpos << metrics_.topo_similarity
                      << " churn=" << std::noshowpos << metrics_.churn << "\n";
        }
    }
}

void SwarmKernel::step() {
    // Select speaker weighted by prestige
    std::vector<double> weights;
    weights.reserve(agents_.size());
    for (const auto& agent : agents_) {
        weights.push_back(agent.w);
    }
    int speaker_idx = weighted_choice(rng_, weights);
    
    // Select hearers
    std::vector<int> hearers = select_hearers(speaker_idx);
    
    // Generate context
    Context ctx(rng_);
    
    // Produce message
    std::vector<std::string> msg = produce(speaker_idx, ctx);
    
    // Interact and learn
    bool success = interact(speaker_idx, hearers, ctx, msg);
    learn_from(speaker_idx, msg, ctx, success);
    for (int hearer_idx : hearers) {
        learn_from(hearer_idx, msg, ctx, success);
    }
    
    // Update prestige
    std::vector<int> all_participants = {speaker_idx};
    all_participants.insert(all_participants.end(), hearers.begin(), hearers.end());
    update_prestige(all_participants, success);
    
    // Maybe mutate
    std::uniform_real_distribution<double> mut_dist(0.0, 1.0);
    if (mut_dist(rng_) < cfg_.mutation_rate) {
        std::uniform_int_distribution<int> agent_dist(0, agents_.size() - 1);
        mutate_agent(agent_dist(rng_));
    }
    
    // Update clusters periodically
    if (t_ % cfg_.cluster_update_freq == 0) {
        update_clusters();
    }
    
    t_++;
}

std::vector<int> SwarmKernel::select_hearers(int speaker_idx) {
    if (social_graph_.find(speaker_idx) == social_graph_.end() || social_graph_[speaker_idx].empty()) {
        // Fallback to random selection
        std::vector<int> candidates;
        for (int i = 0; i < static_cast<int>(agents_.size()); ++i) {
            if (i != speaker_idx) {
                candidates.push_back(i);
            }
        }
        return random_sample(rng_, candidates, std::min(3, static_cast<int>(candidates.size())));
    }
    
    std::vector<int> neighbors = social_graph_[speaker_idx];
    std::vector<double> weights;
    weights.reserve(neighbors.size());
    
    for (int neighbor_idx : neighbors) {
        double similarity = cosine_similarity(agents_[speaker_idx].belief, agents_[neighbor_idx].belief);
        double weight = 1.0 + cfg_.coercion * similarity;
        weights.push_back(weight);
    }
    
    int k = std::min(3, static_cast<int>(neighbors.size()));
    std::vector<int> selected_indices = weighted_sample(rng_, neighbors.size(), weights, k);
    
    std::vector<int> hearers;
    hearers.reserve(selected_indices.size());
    for (int idx : selected_indices) {
        if (idx < static_cast<int>(neighbors.size())) {
            hearers.push_back(neighbors[idx]);
        }
    }
    
    return hearers;
}

std::vector<std::string> SwarmKernel::produce(int agent_idx, const Context& ctx) {
    auto& agent = agents_[agent_idx];
    std::vector<std::string> msg;
    msg.reserve(cfg_.max_message_len);
    
    for (int i = 0; i < cfg_.max_message_len; ++i) {
        std::vector<std::pair<std::string, double>> candidates;
        
        // Add existing associations
        for (const auto& [form, vec] : agent.assoc) {
            double score = dot_product(vec, ctx.vec);
            candidates.emplace_back(form, score);
        }
        
        // Add exploration option
        std::uniform_real_distribution<double> explore_dist(0.0, 1.0);
        if (explore_dist(rng_) < cfg_.exploration_eps || candidates.empty()) {
            auto theonyms = get_theonyms();
            std::uniform_int_distribution<int> theo_dist(0, theonyms.size() - 1);
            std::string new_form = theonyms[theo_dist(rng_)];
            candidates.emplace_back(new_form, 0.5);
        }
        
        if (!candidates.empty()) {
            std::string chosen = softmax_choice(candidates);
            msg.push_back(chosen);
            tokens_.push_back(chosen);
        }
    }
    
    return msg;
}

std::string SwarmKernel::softmax_choice(const std::vector<std::pair<std::string, double>>& items) {
    if (items.empty()) return "";
    
    std::vector<double> exp_scores;
    exp_scores.reserve(items.size());
    double max_score = std::max_element(items.begin(), items.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; })->second;
    
    for (const auto& [form, score] : items) {
        exp_scores.push_back(std::exp(score - max_score));
    }
    
    int chosen_idx = weighted_choice(rng_, exp_scores);
    return items[chosen_idx].first;
}

bool SwarmKernel::interact(int speaker_idx, const std::vector<int>& hearers, 
                          const Context& ctx, const std::vector<std::string>& msg) {
    if (hearers.empty() || msg.empty()) return false;
    
    double total_success = 0.0;
    for (int hearer_idx : hearers) {
        auto& hearer = agents_[hearer_idx];
        double hearer_score = 0.0;
        
        for (const std::string& token : msg) {
            if (hearer.assoc.find(token) != hearer.assoc.end()) {
                hearer_score += dot_product(hearer.assoc[token], ctx.vec);
            }
        }
        
        hearer_score /= msg.size();
        total_success += hearer_score;
    }
    
    double avg_success = total_success / hearers.size();
    return avg_success > cfg_.base_success_thresh;
}

void SwarmKernel::learn_from(int agent_idx, const std::vector<std::string>& msg, 
                            const Context& ctx, bool success) {
    auto& agent = agents_[agent_idx];
    double rate = success ? cfg_.learning_rate : -cfg_.penalty_rate;
    
    for (const std::string& token : msg) {
        if (agent.assoc.find(token) == agent.assoc.end()) {
            agent.assoc[token] = random_unit_vector(rng_, AXES.size());
        }
        
        // Update association
        for (size_t i = 0; i < agent.assoc[token].size(); ++i) {
            agent.assoc[token][i] += rate * ctx.vec[i];
        }
        normalize_vector(agent.assoc[token]);
        
        // Update frequency
        agent.freq[token]++;
    }
}

void SwarmKernel::update_prestige(const std::vector<int>& agent_indices, bool success) {
    double delta = success ? cfg_.prestige_alpha : -cfg_.prestige_alpha;
    
    for (int idx : agent_indices) {
        agents_[idx].w = clamp(agents_[idx].w + delta, 0.1, 2.0);
    }
}

void SwarmKernel::mutate_agent(int agent_idx) {
    auto& agent = agents_[agent_idx];
    
    if (!agent.assoc.empty()) {
        auto it = agent.assoc.begin();
        std::advance(it, rng_() % agent.assoc.size());
        it->second = jitter_vector(it->second, rng_, 0.1);
    }
}

void SwarmKernel::update_clusters() {
    clusters_.clear();
    centroids_.clear();
    
    for (const auto& agent : agents_) {
        if (centroids_.empty()) {
            centroids_.push_back(copy_vector(agent.belief));
            clusters_.push_back({agent.id});
            continue;
        }
        
        std::vector<double> similarities;
        similarities.reserve(centroids_.size());
        for (const auto& centroid : centroids_) {
            similarities.push_back(cosine_similarity(agent.belief, centroid));
        }
        
        auto max_it = std::max_element(similarities.begin(), similarities.end());
        double max_sim = *max_it;
        int best_idx = std::distance(similarities.begin(), max_it);
        
        if (max_sim >= cfg_.cluster_threshold) {
            clusters_[best_idx].push_back(agent.id);
        } else {
            centroids_.push_back(copy_vector(agent.belief));
            clusters_.push_back({agent.id});
        }
    }
    
    // Recalculate centroids
    for (size_t i = 0; i < clusters_.size(); ++i) {
        if (clusters_[i].empty()) continue;
        
        std::vector<double> new_centroid(AXES.size(), 0.0);
        for (int agent_id : clusters_[i]) {
            for (size_t j = 0; j < AXES.size(); ++j) {
                new_centroid[j] += agents_[agent_id].belief[j];
            }
        }
        
        for (double& val : new_centroid) {
            val /= clusters_[i].size();
        }
        normalize_vector(new_centroid);
        centroids_[i] = std::move(new_centroid);
    }
}

void SwarmKernel::update_metrics() {
    metrics_.zipf_slope = calculate_zipf_slope();
    metrics_.heaps_k = calculate_heaps_k();
    metrics_.cond_entropy = calculate_conditional_entropy();
    metrics_.topo_similarity = 0.0; // Simplified for now
    metrics_.churn = 0.0; // Simplified for now
}

double SwarmKernel::calculate_zipf_slope() {
    if (types_.empty()) return 0.0;
    
    std::vector<std::pair<int, std::string>> freq_pairs;
    for (const auto& [type, freq] : types_) {
        freq_pairs.emplace_back(freq, type);
    }
    
    std::sort(freq_pairs.rbegin(), freq_pairs.rend());
    
    if (freq_pairs.size() < 2) return 0.0;
    
    double sum_log_rank = 0.0, sum_log_freq = 0.0;
    double sum_log_rank_sq = 0.0, sum_log_rank_freq = 0.0;
    int n = std::min(static_cast<int>(freq_pairs.size()), 20);
    
    for (int i = 0; i < n; ++i) {
        double log_rank = std::log(i + 1);
        double log_freq = std::log(freq_pairs[i].first);
        
        sum_log_rank += log_rank;
        sum_log_freq += log_freq;
        sum_log_rank_sq += log_rank * log_rank;
        sum_log_rank_freq += log_rank * log_freq;
    }
    
    double denominator = n * sum_log_rank_sq - sum_log_rank * sum_log_rank;
    if (std::abs(denominator) < 1e-10) return 0.0;
    
    return (n * sum_log_rank_freq - sum_log_rank * sum_log_freq) / denominator;
}

double SwarmKernel::calculate_heaps_k() {
    if (tokens_.empty()) return 0.0;
    
    int V = types_.size();
    int N = tokens_.size();
    
    if (N <= 1) return 0.0;
    
    return V / std::pow(N, 0.5); // Simplified Heaps law
}

double SwarmKernel::calculate_conditional_entropy() {
    if (bigrams_.empty()) return 0.0;
    
    double total_entropy = 0.0;
    int total_contexts = 0;
    
    for (const auto& [context, next_counts] : bigrams_) {
        int context_total = 0;
        for (const auto& [next, count] : next_counts) {
            context_total += count;
        }
        
        if (context_total > 0) {
            double context_entropy = 0.0;
            for (const auto& [next, count] : next_counts) {
                double prob = static_cast<double>(count) / context_total;
                context_entropy -= prob * std::log2(prob);
            }
            total_entropy += context_entropy;
            total_contexts++;
        }
    }
    
    return total_contexts > 0 ? total_entropy / total_contexts : 0.0;
}

void SwarmKernel::remove_edge(int a, int b) {
    if (social_graph_.find(a) != social_graph_.end()) {
        remove_item(social_graph_[a], b);
    }
    if (social_graph_.find(b) != social_graph_.end()) {
        remove_item(social_graph_[b], a);
    }
}

bool SwarmKernel::has_edge(int a, int b) const {
    auto it = social_graph_.find(a);
    if (it != social_graph_.end()) {
        const auto& neighbors = it->second;
        return std::find(neighbors.begin(), neighbors.end(), b) != neighbors.end();
    }
    return false;
}
