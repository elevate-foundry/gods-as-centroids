#pragma once

#include <vector>
#include <string>
#include <unordered_map>

class Agent {
public:
    int id;
    std::vector<double> belief;
    double w;
    std::unordered_map<std::string, std::vector<double>> assoc;
    std::unordered_map<std::string, int> freq;

    Agent(int id, std::vector<double> belief);
    ~Agent() = default;
    
    // Make Agent movable
    Agent(const Agent&) = default;
    Agent& operator=(const Agent&) = default;
    Agent(Agent&&) = default;
    Agent& operator=(Agent&&) = default;
};
