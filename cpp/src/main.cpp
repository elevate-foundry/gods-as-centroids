#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include "kernel.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  -c, --config FILE    Configuration JSON file (default: config.json)\n"
              << "  -s, --steps N        Number of simulation steps (default: 5000)\n"
              << "  -l, --label LABEL    Run label for output directory (default: cpp_run)\n"
              << "  -h, --help           Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::string configPath = "config.json";
    int steps = 5000;
    std::string label = "cpp_run";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0) {
            if (i + 1 < argc) {
                configPath = argv[++i];
            } else {
                std::cerr << "Error: --config requires a filename\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--steps") == 0) {
            if (i + 1 < argc) {
                steps = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --steps requires a number\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--label") == 0) {
            if (i + 1 < argc) {
                label = argv[++i];
            } else {
                std::cerr << "Error: --label requires a string\n";
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Error: Unknown option " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "⚡ Gods as Centroids GABM (C++)\n";
    std::cout << "Config: " << configPath << "\n";
    std::cout << "Steps: " << steps << "\n";
    std::cout << "Label: " << label << "\n";

    try {
        // Load configuration
        std::ifstream configFile(configPath);
        if (!configFile.is_open()) {
            std::cerr << "Error: Cannot open config file " << configPath << "\n";
            return 1;
        }

        json configJson;
        configFile >> configJson;
        Config config = configJson.get<Config>();

        // Initialize and run simulation
        SwarmKernel kernel(config);
        kernel.run(steps);

        std::cout << "✅ Simulation complete!\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
