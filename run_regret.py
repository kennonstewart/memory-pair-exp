#!/usr/bin/env python3
"""
Memory-Pair Experiment: Regret Analysis
=======================================

This script evaluates the cumulative regret of different online learning algorithms
on streaming data with various types of drift.
"""

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from src.data_streams import get_dataset_generator
from src.algorithms import get_algorithm


@click.command()
@click.option('--dataset', type=click.Choice(['rotmnist', 'covtype']), required=True,
              help='Dataset to use')
@click.option('--stream', type=click.Choice(['iid', 'drift', 'adv']), required=True,
              help='Stream type')
@click.option('--algo', type=click.Choice(['memorypair', 'sgd', 'adagrad', 'ons']), required=True,
              help='Algorithm to use')
@click.option('--T', type=int, default=100000, help='Number of time steps')
@click.option('--seed', type=int, default=42, help='Random seed')
@click.option('--plot', is_flag=True, help='Generate plot after evaluation')
def main(dataset, stream, algo, t, seed, plot):
    """Run regret evaluation for specified configuration."""
    
    T = t  # Use T internally for consistency with the paper
    
    print(f"Starting experiment: {dataset}_{stream}_{algo}")
    print(f"Parameters: T={T}, seed={seed}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Get data stream generator
    print("Setting up data stream...")
    stream_gen = get_dataset_generator(dataset, stream, seed)
    
    # Get algorithm
    print("Initializing algorithm...")
    # Determine number of features and classes based on dataset
    if dataset == "rotmnist":
        n_features = 784  # 28x28 images
        n_classes = 10
    elif dataset == "covtype":
        n_features = 54  # COVTYPE has 54 features
        n_classes = 7
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    algorithm = get_algorithm(algo, n_features, n_classes, seed)
    
    # Run online learning
    print("Running online learning...")
    regret_data = []
    
    for t, (x, y) in enumerate(tqdm(stream_gen.stream(T), total=T, desc="Processing")):
        # Perform one step
        loss = algorithm.step(x, y)
        
        # Log regret periodically
        if t % 1000 == 0 or t == T - 1:
            regret_data.append({
                'step': t + 1,
                'regret': algorithm.cumulative_regret
            })
    
    # Save results
    results_df = pd.DataFrame(regret_data)
    results_file = f"results/{dataset}_{stream}_{algo}.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"Results saved to {results_file}")
    print(f"Final cumulative regret: {algorithm.cumulative_regret:.2f}")
    
    # Generate plot if requested
    if plot:
        generate_plot(dataset, stream, algo, results_df, T)


def generate_plot(dataset, stream, algo, results_df, T):
    """Generate regret plot with sqrt(T) guideline."""
    
    plt.figure(figsize=(10, 6))
    
    # Plot cumulative regret
    plt.loglog(results_df['step'], results_df['regret'] + 1,  # Add 1 to avoid log(0)
               label=f'{algo.upper()}', linewidth=2)
    
    # Plot sqrt(T) guideline
    steps = np.array(results_df['step'])
    sqrt_T = np.sqrt(steps)
    
    # Scale the sqrt(T) guideline to be meaningful
    final_regret = max(results_df['regret'].iloc[-1], 1.0)
    plt.loglog(steps, sqrt_T * (final_regret / sqrt_T[-1]), 
               '--', alpha=0.7, label='√T guideline')
    
    plt.xlabel('Time step (T)')
    plt.ylabel('Cumulative regret')
    plt.title(f'Cumulative Regret: {dataset.upper()} ({stream}) - {algo.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_file = f"results/{dataset}_{stream}_{algo}_plot.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {plot_file}")


def run_all_experiments():
    """Run all combinations of experiments."""
    datasets = ['rotmnist', 'covtype']
    streams = ['iid', 'drift', 'adv']
    algorithms = ['memorypair', 'sgd', 'adagrad', 'ons']
    
    T = 10000  # Smaller T for comprehensive testing
    seed = 42
    
    print("Running comprehensive experiments...")
    
    for dataset in datasets:
        for stream in streams:
            for algo in algorithms:
                print(f"\n{'='*60}")
                print(f"Running: {dataset}_{stream}_{algo}")
                print(f"{'='*60}")
                
                try:
                    # Import click context for programmatic invocation
                    from click.testing import CliRunner
                    
                    runner = CliRunner()
                    result = runner.invoke(main, [
                        '--dataset', dataset,
                        '--stream', stream,
                        '--algo', algo,
                        '--T', str(T),
                        '--seed', str(seed),
                        '--plot'
                    ])
                    
                    if result.exit_code != 0:
                        print(f"Error in {dataset}_{stream}_{algo}: {result.output}")
                    
                except Exception as e:
                    print(f"Error running {dataset}_{stream}_{algo}: {e}")
    
    print("\nGenerating summary plots...")
    generate_summary_plots(datasets, streams, algorithms)


def generate_summary_plots(datasets, streams, algorithms):
    """Generate summary comparison plots."""
    
    for dataset in datasets:
        for stream in streams:
            plt.figure(figsize=(12, 8))
            
            for algo in algorithms:
                try:
                    results_file = f"results/{dataset}_{stream}_{algo}.csv"
                    if os.path.exists(results_file):
                        df = pd.read_csv(results_file)
                        plt.loglog(df['step'], df['regret'], 
                                  label=f'{algo.upper()}', linewidth=2)
                except Exception as e:
                    print(f"Error loading {results_file}: {e}")
            
            # Add sqrt(T) guideline
            if len(plt.gca().lines) > 0:
                max_T = max([max(line.get_xdata()) for line in plt.gca().lines])
                steps = np.logspace(2, np.log10(max_T), 100)
                sqrt_T = np.sqrt(steps)
                plt.loglog(steps, sqrt_T * 10, '--', alpha=0.7, label='√T guideline')
            
            plt.xlabel('Time step (T)')
            plt.ylabel('Cumulative regret')
            plt.title(f'Algorithm Comparison: {dataset.upper()} ({stream})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save summary plot
            summary_file = f"results/summary_{dataset}_{stream}.png"
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Summary plot saved to {summary_file}")


if __name__ == "__main__":
    # Check if running with --all flag
    import sys
    if '--all' in sys.argv:
        run_all_experiments()
    else:
        main()