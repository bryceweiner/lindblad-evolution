# lindblad-evolution

[![PyPI version](https://badge.fury.io/py/lindblad-evolution.svg)](https://pypi.org/project/lindblad-evolution/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A differentiable Lindblad master equation solver with full Apple Silicon (MPS) support for physics research. Implements open quantum system dynamics with MPS-compatible alternatives to eigendecomposition.

## ✨ Key Features

- **🔬 Differentiable Lindblad Evolution**: Full gradient flow through quantum master equations
- **🍎 Apple Silicon Optimized**: Native MPS support with differentiable eigendecomposition alternatives
- **🧠 Physics-Informed AI Ready**: Seamless integration with neural networks and optimization frameworks
- **⚡ Hardware Acceleration**: Automatic CUDA/MPS/CPU device selection with intelligent fallbacks
- **🔧 Modular Hamiltonian Support**: Geometric KMS conditions and modular flow operators
- **📊 Comprehensive Metrics**: Trace preservation, purity, and Von Neumann entropy tracking

## 🚀 Installation

```bash
pip install lindblad-evolution
```

**Requirements:**
- Python ≥ 3.9
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.21.0

## 📖 Quick Start

```python
import torch
from lindblad_evolution import LindbladEvolutionEngine

# Create a quantum system with 32 states and 8 Lindblad operators
engine = LindbladEvolutionEngine(system_dim=32, n_lindblad_ops=8)

# Prepare a quantum state (normalized complex vector)
psi = torch.randn(32, dtype=torch.complex64)
psi = psi / psi.norm()

# Evolve the quantum state
evolved_state, metrics = engine(psi.unsqueeze(0))

print(f"Trace: {metrics['trace']:.4f}")
print(f"Purity: {metrics['purity']:.4f}")
print(f"Entropy: {metrics['von_neumann_entropy']:.4f}")
```

## 🔧 API Reference

### LindbladEvolutionEngine

The main evolution engine for open quantum system dynamics.

```python
engine = LindbladEvolutionEngine(
    system_dim=496,           # Quantum system dimension
    n_lindblad_ops=8,         # Number of Lindblad jump operators
    device=None,              # Auto-detect optimal device (cuda/mps/cpu)
    use_modular_hamiltonian=True  # Include geometric modular Hamiltonian
)
```

**Methods:**
- `forward(input_state, evolution_time=1.0, n_steps=10)`: Evolve quantum state
- `evolve_density_matrix(rho, dt, n_steps)`: Direct density matrix evolution

**Returns:** `(evolved_state, metrics_dict)`
- `evolved_state`: Shape `(batch_size, system_dim)`
- `metrics`: Dictionary with `trace`, `purity`, `von_neumann_entropy`, `total_dissipation`

### ModularHamiltonian

Geometric modular Hamiltonian with KMS conditions.

```python
from lindblad_evolution import ModularHamiltonian

mod_h = ModularHamiltonian(
    system_dim=32,
    diamond_radius=1.0,
    device="mps"  # Device for computations
)

# Apply modular flow
rho_evolved = mod_h.modular_flow(rho, modular_time=0.1)
```

### LindbladOperator

Individual jump operators for dissipative dynamics.

```python
from lindblad_evolution import LindbladOperator

# Create a Lindblad operator
L = torch.randn(32, 32, dtype=torch.complex64)
jump_op = LindbladOperator("dephasing", L, rate=0.1)

# Apply to density matrix
dissipation = jump_op.apply(rho)
```

### Utility Functions

```python
from lindblad_evolution.utils import get_device, get_device_info

# Get optimal device
device = get_device()  # Auto-detects cuda -> mps -> cpu

# Get device availability info
info = get_device_info()
print(info)
# {'cuda_available': False, 'mps_available': True, 'recommended': 'mps'}
```

### Differentiable Alternatives

MPS-compatible differentiable alternatives to eigendecomposition:

```python
from lindblad_evolution.engine import (
    power_iteration,
    soft_psd_projection,
    differentiable_von_neumann_entropy
)

# Dominant eigenvector (differentiable)
eigenvec = power_iteration(hermitian_matrix, n_iter=20)

# PSD projection with straight-through estimator
rho_psd = soft_psd_projection(rho_with_neg_eigenvals)

# Approximate Von Neumann entropy
entropy = differentiable_von_neumann_entropy(density_matrix)
```

## 🎯 Use Cases

- **Physics-Informed Neural Networks**: Differentiable quantum dynamics in PINNs
- **Quantum Machine Learning**: Variational quantum algorithms with dissipation
- **Quantum Control**: Optimal control of open quantum systems
- **Quantum Simulation**: Efficient simulation of decoherent quantum evolution
- **Thermodynamic Optimization**: Entropy and information flow optimization

## 🔬 Mathematical Framework

The Lindblad master equation describes open quantum system evolution:

```
dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
```

Where:
- `H`: System Hamiltonian (learnable)
- `L_k`: Lindblad jump operators (learnable)
- `γ_k`: Dissipation rates (learnable)
- `ρ`: Density matrix

### MPS-Compatible Operations

For Apple Silicon compatibility, eigendecomposition operations use:

1. **Power Iteration**: Differentiable dominant eigenvector extraction
2. **Soft PSD Projection**: Straight-through estimator for positive semidefinite projection
3. **Series Approximation**: Von Neumann entropy via matrix logarithm series

## 🏗️ Architecture

```
lindblad-evolution/
├── engine.py          # Core Lindblad engine & modular Hamiltonian
├── utils.py           # Device management & MPS optimization
└── __init__.py        # Public API exports
```

## 🤝 Contributing

Contributions welcome! This package is designed for physics-informed AI research.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Repository**: https://github.com/bryceweiner/lindblad-evolution
- **PyPI**: https://pypi.org/project/lindblad-evolution/
- **Issues**: https://github.com/bryceweiner/lindblad-evolution/issues

---

*Built for physics research requiring differentiable open quantum system dynamics.*
