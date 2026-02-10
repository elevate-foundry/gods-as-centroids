use clap::{Arg, Command};
use gods_as_centroids::{Config, SwarmKernel};
use serde_json;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Gods as Centroids GABM")
        .version("0.1.0")
        .author("Rust Port")
        .about("Generative Agent-Based Model of Protolanguage Evolution")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration JSON file")
                .default_value("config.json"),
        )
        .arg(
            Arg::new("steps")
                .short('s')
                .long("steps")
                .value_name("N")
                .help("Number of simulation steps")
                .default_value("5000"),
        )
        .arg(
            Arg::new("label")
                .short('l')
                .long("label")
                .value_name("LABEL")
                .help("Run label for output directory")
                .default_value("rust_run"),
        )
        .get_matches();

    let config_path = matches.get_one::<String>("config").unwrap();
    let steps: usize = matches.get_one::<String>("steps").unwrap().parse()?;
    let label = matches.get_one::<String>("label").unwrap();

    // Load configuration
    let config_str = fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&config_str)?;

    println!("🦀 Gods as Centroids GABM (Rust)");
    println!("Config: {}", config_path);
    println!("Steps: {}", steps);
    println!("Label: {}", label);

    // Initialize and run simulation
    let mut kernel = SwarmKernel::new(config);
    kernel.run(steps);

    println!("✅ Simulation complete!");
    Ok(())
}
