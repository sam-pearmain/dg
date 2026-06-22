use std::{fs, path::PathBuf};

use anyhow::{Ok, Result, anyhow};

use clap::{Parser, Subcommand};
use dg::config::Config;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(short, long)]
    progress: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
#[command(version, about)]
enum Commands {
    /// Run
    Run {
        #[arg(long, value_name = "MESH_FILE")]
        mesh: PathBuf,
        #[arg(long, value_name = "CONFIG_FILE")]
        config: PathBuf,
    },
    /// Launch the auto-config setup
    Setup {
        #[arg(long, value_name = "MESH_FILE")]
        mesh: PathBuf,
    },
}

fn main() -> Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Run { mesh, config } => {
            let universe = mpi::initialize().ok_or(anyhow!("failed to initialise mpi"))?;
            
            // read and parse the config file
            let config_file_contents = fs::read_to_string(config)
                .map_err(|e| anyhow!("failed to read config file: {e}"))?;
            let config: Config = toml::from_str(&config_file_contents)
                .map_err(|e| anyhow!("{e}"))?;

            // construct a dyn system
            
        }
        Commands::Setup { mesh } => todo!(),
    }

    Ok(())
}
