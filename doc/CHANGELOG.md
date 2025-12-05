# Changelog

All notable changes to UMNN will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0] - 2025-12-05

### Critical Fixes
- **Fixed backward pass gradient computation** in `NeuralIntegral` and `ParallelNeuralIntegral`
  - Corrected step size calculation from `x/nb_steps` to `(x - x0)/nb_steps`
  - This bug was causing incorrect gradients during backpropagation
  - Affects: `models/UMNN/NeuralIntegral.py:86` and `models/UMNN/ParallelNeuralIntegral.py:105`
- **Fixed missing return parameter** in `ParallelNeuralIntegral.backward()` for `inv_f` parameter

### Added
- **JIT Compilation Support**: Models can now be compiled with `torch.jit.script()` and `torch.jit.trace()`
  - Refactored `IntegrandNetwork.forward()` for JIT compatibility
  - Enables faster inference in production environments
- **Comprehensive Test Suite**:
  - `test_jit.py`: Tests for JIT compatibility, backward passes, and performance benchmarks
  - `test_numerical_validation.py`: Validates gradient correctness, integral convergence, and fitting capabilities
- **Development Environment**:
  - `environment.yml`: Complete conda/micromamba environment specification
  - `setup_env.sh`: Automated environment setup script
  - `DEV_SETUP.md`: Development setup documentation
- **Documentation**:
  - `FIXES_SUMMARY.md`: Detailed explanation of all changes
  - `PUBLISHING_GUIDE.md`: Complete guide for publishing to PyPI and conda-forge
  - `CHANGELOG.md`: This file

### Changed
- Improved `IntegrandNetwork.forward()` implementation for better clarity and JIT compatibility
- Updated Python requirement to >=3.8 (recommended, though >=3.6 still works)
- Enhanced code comments and documentation throughout

### Testing
- All backward pass tests pass successfully
- Integral convergence verified (error < 1e-4 with 200 quadrature steps)
- Gradient correctness verified through finite difference comparison
- Monotonic function fitting demonstrated (y = x³)
- JIT compilation works for both scripting and tracing

### Performance
- No performance regression
- JIT compilation provides faster inference when enabled
- ParallelNeuralIntegral maintains speedup over sequential NeuralIntegral

## [1.71] - Previous Release

### Changed
- Updated metadata for PyPI and conda-forge consistency
- Build system migration to hatchling
- Added required dependencies specification

## [1.0] - Original Release

### Added
- Initial implementation of Unconstrained Monotonic Neural Networks
- NeuralIntegral with Clenshaw-Curtis quadrature
- ParallelNeuralIntegral for faster computation
- UMNNMAF for autoregressive flows
- Example scripts for toy experiments, UCI datasets, MNIST, and VAE

---

## Migration Guide: 1.x to 2.0

### Breaking Changes
None - version 2.0 is backward compatible with 1.x

### Recommended Actions
1. **Update immediately** if you're training models - the gradient fix is critical
2. **Run tests** to verify your code works with the fixes
3. **Consider JIT compilation** for production deployments

### Code Changes Required
None - the API remains unchanged. However, you may see:
- Better training convergence (due to correct gradients)
- Different trained model weights (due to correct gradient flow)

### Testing Your Migration
```python
# Test that backward pass works
import torch
from models.UMNN import UMNNMAF, EmbeddingNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UMNNMAF(
    net=EmbeddingNetwork(10, [50, 50], [50, 50]),
    input_size=10,
    nb_steps=20,
    device=device
)

x = torch.randn(10, 10, requires_grad=True, device=device)
z = model.forward(x)
loss = z.sum()
loss.backward()

assert x.grad is not None, "Backward pass failed!"
print("✓ Migration successful!")
```

## Support

- **Issues**: https://github.com/AWehenkel/UMNN/issues
- **Discussions**: https://github.com/AWehenkel/UMNN/discussions
- **Email**: antoine.wehenkel@gmail.com

## Citation

If you use UMNN v2.0 in your research, please cite:

```bibtex
@inproceedings{wehenkel2019unconstrained,
  title={Unconstrained monotonic neural networks},
  author={Wehenkel, Antoine and Louppe, Gilles},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1543--1553},
  year={2019}
}
```

## Acknowledgments

- Thanks to the PyTorch team for JIT compilation support
- Thanks to all users who reported issues and provided feedback
- Special thanks to contributors of bug reports and feature requests
