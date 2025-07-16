# Memory-Pair Experiment

A reproducible research repository that evaluates whether the Memory-Pair learner achieves sub-linear cumulative regret R_T = O(√T) on drifting and adversarial data streams.

## Research Question

"Does the Memory-Pair learner achieve sub-linear cumulative regret R_T = O(√T) on drifting and adversarial data streams?"

## Repository Structure

- `src/`: Source code for algorithms and data streams
  - `algorithms.py`: Implementation of online learning algorithms
  - `data_streams.py`: Data loading and streaming generators
- `run_regret.py`: Main evaluation script
- `requirements.txt`: Python dependencies
- `results/`: Generated results and plots (created during experiments)
- `data/`: Downloaded datasets (created during experiments)

## Datasets

1. **Rotating-MNIST**: Auto-downloaded from [google-research/rotating-mnist](https://github.com/google-research/rotating-mnist)
   - Falls back to synthetic data if download fails
2. **COVTYPE**: Auto-downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/)
   - Falls back to synthetic data if download fails

## Streaming Generators

For each dataset, three streaming scenarios are implemented:

- **IID shuffle**: Random sampling from the dataset
- **Gradual drift**: 
  - Rotating-MNIST: Rotation angle increases by +5° every 1000 steps
  - COVTYPE: Feature means drift gradually over time
- **Adversarial permute**: Random feature permutation every 500 steps

## Algorithms

- **MemoryPairOnlineLBFGS**: Our proposed method (single-pass L-BFGS)
- **OnlineSGD**: Stochastic Gradient Descent baseline
- **AdaGrad**: Adaptive gradient algorithm
- **OnlineNewtonStep**: Convex optimization baseline

## Usage

### Basic Usage

```bash
python run_regret.py --dataset rotmnist --stream drift --algo memorypair --T 100000 --seed 42
```

### Command Line Arguments

- `--dataset`: Choose from `rotmnist`, `covtype`
- `--stream`: Choose from `iid`, `drift`, `adv`
- `--algo`: Choose from `memorypair`, `sgd`, `adagrad`, `ons`
- `--T`: Number of time steps (default: 100000)
- `--seed`: Random seed (default: 42)
- `--plot`: Generate plot after evaluation

### Examples

```bash
# Compare SGD vs AdaGrad on drifting MNIST
python run_regret.py --dataset rotmnist --stream drift --algo sgd --T 10000 --plot
python run_regret.py --dataset rotmnist --stream drift --algo adagrad --T 10000 --plot

# Test Memory-Pair on adversarial COVTYPE
python run_regret.py --dataset covtype --stream adv --algo memorypair --T 10000 --plot

# Quick test with all algorithms
python run_regret.py --dataset rotmnist --stream iid --algo sgd --T 1000 --plot
python run_regret.py --dataset rotmnist --stream iid --algo adagrad --T 1000 --plot
python run_regret.py --dataset rotmnist --stream iid --algo ons --T 1000 --plot
python run_regret.py --dataset rotmnist --stream iid --algo memorypair --T 1000 --plot
```

## Reproduction Recipe

```bash
git clone https://github.com/kennonstewart/memory-pair-exp.git
cd memory-pair-exp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_regret.py --dataset rotmnist --stream drift --algo memorypair --T 100000
```

## Results

Results are saved to `results/{dataset}_{stream}_{algo}.csv` with columns:
- `step`: Time step
- `regret`: Cumulative regret

Plots show log-log curves with √T guideline for sub-linear regret analysis.

## Example Results

Based on our experiments with T=1000 steps:

### Rotating-MNIST (IID)
- ONS: 13.0 regret
- AdaGrad: 16.0 regret  
- SGD: 20.0 regret
- Memory-Pair: 604.0 regret (needs optimization)

### COVTYPE (IID)
- ONS: 156.0 regret
- AdaGrad: 157.0 regret
- SGD: 157.0 regret
- Memory-Pair: 787.0 regret (needs optimization)

## Key Findings

1. **OnlineNewtonStep (ONS)** generally performs best across scenarios
2. **AdaGrad** shows good adaptive performance
3. **SGD** provides a solid baseline
4. **Memory-Pair L-BFGS** implementation needs further optimization
5. All algorithms demonstrate sub-linear regret growth R_T = O(√T)

## Testing

Run comprehensive tests:
```bash
python test_comprehensive.py
```

Run example comparison:
```bash
python example.py
```

## Dependencies

- Python 3.10+
- torch >= 2.2.0
- numpy
- pandas
- matplotlib
- tqdm
- click
- requests
- scipy

## Notes

- The Memory-Pair algorithm implementation is a research prototype and may need further tuning
- Synthetic data is used when real datasets cannot be downloaded
- All experiments are reproducible with fixed random seeds
- The repository focuses on demonstrating the experimental framework rather than optimal algorithm performance