use std::path::PathBuf;

use anyhow::{Ok, Result, anyhow};

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
struct Cli {
    #[arg(short, long)]
    progress: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run
    Run {
        #[arg(long)]
        mesh: PathBuf,
        #[arg(long)]
        config: PathBuf,
    },
    /// Launch the auto-config setup environment
    Setup {
        #[arg(long)]
        mesh: PathBuf,
    },
}

fn main() -> Result<()> {
    let args = Cli::parse();

    match args.command {
        Commands::Run { mesh, config } => {
            let universe = mpi::initialize().ok_or(anyhow!("failed to initialise mpi"))?;
            println!("parse the config");
            println!("initialise a dyn solver");
            println!("run the solver");
        }
        Commands::Setup { mesh } => todo!(),
    }

    Ok(())
}
