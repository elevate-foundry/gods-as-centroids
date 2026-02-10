#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <random>

// Utility functions for vector operations and data structures

// Vector operations
void normalize_vector(std::vector<double>& vec);
std::vector<double> random_unit_vector(std::mt19937& rng, int dim);
std::vector<double> jitter_vector(const std::vector<double>& base, std::mt19937& rng, double jitter);
std::vector<double> copy_vector(const std::vector<double>& vec);
double dot_product(const std::vector<double>& a, const std::vector<double>& b);
double cosine_similarity(const std::vector<double>& a, const std::vector<double>& b);

// Sampling functions
int weighted_choice(std::mt19937& rng, const std::vector<double>& weights);
std::vector<int> weighted_sample(std::mt19937& rng, int n, const std::vector<double>& weights, int k);
std::vector<int> random_sample(std::mt19937& rng, const std::vector<int>& items, int k);

// Data structures
std::unordered_map<std::string, std::vector<double>> get_deity_priors();
std::vector<std::string> get_theonyms();

// Math utilities
double clamp(double value, double min, double max);
double sum(const std::vector<double>& vec);
template<typename T>
void remove_item(std::vector<T>& vec, const T& item);

template<typename T>
void remove_item(std::vector<T>& vec, const T& item) {
    vec.erase(std::remove(vec.begin(), vec.end(), item), vec.end());
}
