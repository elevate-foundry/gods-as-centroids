package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
)

func main() {
	var configPath = flag.String("config", "config.json", "Configuration JSON file")
	var steps = flag.Int("steps", 5000, "Number of simulation steps")
	var label = flag.String("label", "go_run", "Run label for output directory")
	flag.Parse()

	fmt.Println("🐹 Gods as Centroids GABM (Go)")
	fmt.Printf("Config: %s\n", *configPath)
	fmt.Printf("Steps: %d\n", *steps)
	fmt.Printf("Label: %s\n", *label)

	// Load configuration
	configData, err := os.ReadFile(*configPath)
	if err != nil {
		log.Fatal(err)
	}

	var config Config
	if err := json.Unmarshal(configData, &config); err != nil {
		log.Fatal(err)
	}

	// Initialize and run simulation
	kernel := NewSwarmKernel(config)
	kernel.Run(*steps)

	fmt.Println("✅ Simulation complete!")
}
