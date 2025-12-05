# UMNN Development Environment

## Quick Setup

To set up the development environment with micromamba:

```bash
# Method 1: Use the setup script
./setup_env.sh

# Method 2: Manual setup
micromamba env create -f environment.yml -y
micromamba activate umnn-dev
```

## Environment Details

The `umnn-dev` environment includes:
- Python >= 3.8
- PyTorch >= 2.0
- NumPy
- Matplotlib
- SciPy
- Jupyter
- Pytest (for testing)
- Black & Flake8 (for code formatting)
- TensorBoard (for experiment tracking)

## Testing

Run the test suite to verify JIT compatibility and backward pass:

```bash
python test_jit.py
```

This test script will:
1. Test backward pass correctness for NeuralIntegral and ParallelNeuralIntegral
2. Test the full UMNN model forward and backward passes
3. Test JIT compatibility (tracing and scripting)
4. Benchmark performance between sequential and parallel implementations

## Installing micromamba

If you don't have micromamba installed:

```bash
# For macOS (ARM64)
curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj bin/micromamba

# For macOS (Intel)
curl -Ls https://micro.mamba.pm/api/micromamba/osx-64/latest | tar -xvj bin/micromamba

# For Linux
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
```

Then follow the initialization instructions.

## Recent Fixes

### Backward Pass Fix (2024)
Fixed incorrect step size calculation in the backward methods of both `NeuralIntegral` and `ParallelNeuralIntegral`:
- Changed `x/nb_steps` to `(x - x0)/nb_steps` in models/UMNN/NeuralIntegral.py:86
- Changed `x/nb_steps` to `(x - x0)/nb_steps` in models/UMNN/ParallelNeuralIntegral.py:105
- Added missing return value for `inv_f` parameter in ParallelNeuralIntegral.py:109-112

These fixes ensure correct gradient computation during backpropagation.
