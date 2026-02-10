#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <memory>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// Semantic axes for deity representation
extern const std::vector<std::string> AXES;

// Configuration structure
struct Config {
    int n = 50;
    int steps_per_generation = 2500;
    int max_message_len = 3;
    double learning_rate = 0.08;
    double penalty_rate = 0.02;
    double prestige_alpha = 0.20;
    int ritual_period = 50;
    double ritual_bonus = 0.10;
    double base_success_thresh = 0.58;
    double mutation_rate = 0.08;
    double exploration_eps = 0.10;
    int generation_mix_k = 3;
    uint64_t seed = 42;
    int topo_window = 200;
    bool use_deity_priors = true;
    double belief_influence = 0.15;
    double coercion = 0.0;
    std::string social_network = "small_world";
    int social_k = 4;
    double social_p = 0.1;
    int cluster_update_freq = 100;
    double cluster_threshold = 0.4;
};

// JSON serialization for Config
void to_json(json& j, const Config& c);
void from_json(const json& j, Config& c);

// Context structure
struct Context {
    std::string task;
    std::string role;
    std::string place;
    std::string tod;
    std::vector<double> vec;
    
    Context(std::mt19937& rng);
};

// Forward declarations
#include "agent.h"

// Metrics structure
struct Metrics {
    double zipf_slope = 0.0;
    double heaps_k = 0.0;
    double cond_entropy = 0.0;
    double topo_similarity = 0.0;
    double churn = 0.0;
};

// Main simulation kernel
class SwarmKernel {
private:
    Config cfg_;
    std::mt19937 rng_;
    std::vector<Agent> agents_;
    int t_ = 0;
    int gen_ = 0;
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int> types_;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> bigrams_;
    std::string last_token_;
    std::unordered_map<int, std::string> pref_form_;
    Metrics metrics_;
    std::unordered_map<int, std::vector<int>> social_graph_;
    std::vector<std::vector<int>> clusters_;
    std::vector<std::vector<double>> centroids_;

public:
    explicit SwarmKernel(const Config& cfg);
    ~SwarmKernel() = default;

    void run(int steps);

private:
    void init_agents();
    void build_social_network();
    void step();
    std::vector<int> select_hearers(int speaker_idx);
    std::vector<std::string> produce(int agent_idx, const Context& ctx);
    std::string softmax_choice(const std::vector<std::pair<std::string, double>>& items);
    bool interact(int speaker_idx, const std::vector<int>& hearers, 
                  const Context& ctx, const std::vector<std::string>& msg);
    void learn_from(int agent_idx, const std::vector<std::string>& msg, 
                    const Context& ctx, bool success);
    void update_prestige(const std::vector<int>& agent_indices, bool success);
    void mutate_agent(int agent_idx);
    void update_clusters();
    void update_metrics();
    
    // Metrics calculations
    double calculate_zipf_slope();
    double calculate_heaps_k();
    double calculate_conditional_entropy();
    
    // Utility methods
    void remove_edge(int a, int b);
    bool has_edge(int a, int b) const;
};
