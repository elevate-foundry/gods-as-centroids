#include "utils.h"
#include <cmath>
#include <algorithm>
#include <numeric>

// Vector operations
void normalize_vector(std::vector<double>& vec) {
    double norm = 0.0;
    for (double v : vec) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (double& v : vec) {
            v /= norm;
        }
    }
}

std::vector<double> random_unit_vector(std::mt19937& rng, int dim) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> vec(dim);
    for (int i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    normalize_vector(vec);
    return vec;
}

std::vector<double> jitter_vector(const std::vector<double>& base, std::mt19937& rng, double jitter) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> vec(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        vec[i] = base[i] + dist(rng) * jitter;
    }
    normalize_vector(vec);
    return vec;
}

std::vector<double> copy_vector(const std::vector<double>& vec) {
    return std::vector<double>(vec);
}

double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double cosine_similarity(const std::vector<double>& a, const std::vector<double>& b) {
    double dot = dot_product(a, b);
    double norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a > 0 && norm_b > 0) {
        return dot / (norm_a * norm_b);
    }
    return 0.0;
}

// Sampling functions
int weighted_choice(std::mt19937& rng, const std::vector<double>& weights) {
    double total = sum(weights);
    std::uniform_real_distribution<double> dist(0.0, total);
    double r = dist(rng);
    
    for (size_t i = 0; i < weights.size(); ++i) {
        r -= weights[i];
        if (r <= 0) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(weights.size() - 1);
}

std::vector<int> weighted_sample(std::mt19937& rng, int n, const std::vector<double>& weights, int k) {
    std::vector<int> result;
    result.reserve(k);
    double total = sum(weights);
    std::uniform_real_distribution<double> dist(0.0, total);
    
    for (int i = 0; i < k && result.size() < static_cast<size_t>(n); ++i) {
        double r = dist(rng);
        for (int j = 0; j < static_cast<int>(weights.size()); ++j) {
            r -= weights[j];
            if (r <= 0) {
                result.push_back(j);
                break;
            }
        }
    }
    return result;
}

std::vector<int> random_sample(std::mt19937& rng, const std::vector<int>& items, int k) {
    if (k >= static_cast<int>(items.size())) {
        return items;
    }
    
    std::vector<int> indices(items);
    std::vector<int> result;
    result.reserve(k);
    
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> dist(0, static_cast<int>(indices.size()) - 1);
        int idx = dist(rng);
        result.push_back(indices[idx]);
        indices.erase(indices.begin() + idx);
    }
    return result;
}

// Data structures
std::unordered_map<std::string, std::vector<double>> get_deity_priors() {
    std::unordered_map<std::string, std::vector<double>> priors = {
        {"zeus", {0.9, 0.8, 0.3, 0.7, 0.6, 0.9, 0.2, 0.8, 0.1, 0.4, 0.3, 0.8}},
        {"odin", {0.8, 0.7, 0.4, 0.6, 0.9, 0.7, 0.1, 0.9, 0.8, 0.3, 0.2, 0.5}},
        {"amun", {0.9, 0.9, 0.6, 0.8, 0.8, 0.8, 0.3, 0.2, 0.1, 0.9, 0.1, 0.9}},
        {"marduk", {0.9, 0.6, 0.5, 0.9, 0.7, 0.9, 0.1, 0.8, 0.3, 0.7, 0.1, 0.9}},
        {"indra", {0.8, 0.5, 0.4, 0.7, 0.6, 0.9, 0.2, 0.9, 0.2, 0.3, 0.4, 0.6}},
        {"shango", {0.7, 0.4, 0.3, 0.8, 0.5, 0.8, 0.1, 0.7, 0.2, 0.2, 0.6, 0.5}},
        {"kami", {0.3, 0.8, 0.8, 0.4, 0.7, 0.4, 0.6, 0.1, 0.1, 0.5, 0.9, 0.8}},
        {"manitou", {0.2, 0.9, 0.9, 0.3, 0.8, 0.3, 0.7, 0.1, 0.2, 0.6, 0.9, 0.4}},
    };
    
    // Normalize all vectors
    for (auto& [name, vector] : priors) {
        normalize_vector(vector);
    }
    
    return priors;
}

std::vector<std::string> get_theonyms() {
    return {
        "zeus", "odin", "amun", "marduk", "indra", "shango", "kami", "manitou",
        "apollo", "freya", "ptah", "ishtar", "perun", "teotl", "nut", "hades",
        "thor", "isis", "ra", "quetzal", "tyr", "bast", "lugh", "brigid",
        "taranis", "nana", "enar", "yah", "baal"
    };
}

// Math utilities
double clamp(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

double sum(const std::vector<double>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.0);
}
