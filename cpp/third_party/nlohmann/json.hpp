// Placeholder for nlohmann/json.hpp
// In a real implementation, this would be the full nlohmann/json header
// For now, we'll create a minimal stub to allow compilation

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace nlohmann {
    class json {
    public:
        json() = default;
        json(const std::unordered_map<std::string, int>& m) {}
        
        template<typename T>
        T get() const { return T{}; }
        
        json& at(const std::string& key) { return *this; }
        const json& at(const std::string& key) const { return *this; }
        
        template<typename T>
        void get_to(T& val) const {}
        
        friend std::istream& operator>>(std::istream& is, json& j) { return is; }
    };
}
