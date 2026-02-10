#include "agent.h"

Agent::Agent(int id, std::vector<double> belief) 
    : id(id), belief(std::move(belief)), w(1.0) {
}
