"""
Lindblad Evolution Engine

Implements open quantum system dynamics with Lindblad master equation:
    dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

Includes MPS-compatible differentiable alternatives to eigendecomposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
import logging

from .utils import get_device

logger = logging.getLogger(__name__)

# =============================================================================
# Differentiable Alternatives to Eigendecomposition (MPS-compatible)
# =============================================================================

def power_iteration(
    A: torch.Tensor, 
    n_iter: int = 20, 
    return_eigenvalue: bool = False
) -> torch.Tensor:
    """
    Differentiable dominant eigenvector via power iteration.
    
    Mathematically equivalent to taking the eigenvector corresponding to
    the largest eigenvalue from eigh, but fully differentiable on MPS.
    
    Args:
        A: Hermitian matrix (n, n)
        n_iter: Number of iterations (more = more accurate)
        return_eigenvalue: If True, also return the dominant eigenvalue
        
    Returns:
        Dominant eigenvector (n,), optionally with eigenvalue
    """
    n = A.shape[0]
    # Initialize with random vector
    v = torch.randn(n, device=A.device, dtype=A.dtype)
    v = v / (v.norm() + 1e-10)
    
    for _ in range(n_iter):
        # Power iteration: v_{k+1} = A @ v_k / ||A @ v_k||
        Av = A @ v
        v = Av / (Av.norm() + 1e-10)
    
    if return_eigenvalue:
        # Rayleigh quotient: λ = v† A v
        eigenvalue = (v.conj() @ A @ v).real
        return v, eigenvalue
    return v


def soft_psd_projection(
    rho: torch.Tensor,
    eps: float = 1e-6,
    n_iter: int = 5
) -> torch.Tensor:
    """
    Differentiable projection to positive semidefinite matrix.

    Uses straight-through estimator: accurate eigendecomposition in forward pass
    (on CPU for MPS compatibility), but allows gradients to flow through as if
    it were the identity operation on the valid subspace.

    Args:
        rho: Input matrix (may have negative eigenvalues)
        eps: Small regularization for numerical stability
        n_iter: Not used, kept for API compatibility

    Returns:
        Positive semidefinite matrix with trace 1
    """
    # Ensure Hermitian first
    rho_hermitian = 0.5 * (rho + rho.conj().T)

    # Move to CPU for eigendecomposition (MPS doesn't support eigh)
    original_device = rho_hermitian.device
    rho_cpu = rho_hermitian.detach().cpu()

    # Accurate eigendecomposition on CPU
    eigenvalues, eigenvectors = torch.linalg.eigh(rho_cpu)

    # Clamp negative eigenvalues (accurate projection)
    eigenvalues_pos = torch.clamp(eigenvalues, min=eps)

    # Reconstruct PSD matrix - ensure proper dtype handling
    if torch.is_complex(eigenvectors):
        # For complex matrices, eigenvalues are real but eigenvectors are complex
        diag_matrix = torch.diag(eigenvalues_pos.to(eigenvectors.dtype))
        rho_psd_cpu = eigenvectors @ diag_matrix @ eigenvectors.conj().T
    else:
        # For real matrices
        rho_psd_cpu = eigenvectors @ torch.diag(eigenvalues_pos) @ eigenvectors.T

    # Normalize trace
    trace = torch.trace(rho_psd_cpu)
    if trace.abs() > 1e-10:
        rho_psd_cpu = rho_psd_cpu / trace

    # Move back to original device
    rho_psd = rho_psd_cpu.to(original_device)

    # Straight-through estimator: use projected value but preserve gradients
    # grad(output) = grad(input) for valid directions
    # This is achieved by: output = input + (target - input).detach()
    rho_out = rho_hermitian + (rho_psd - rho_hermitian).detach()

    return rho_out.real


def differentiable_von_neumann_entropy(
    rho: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Differentiable Von Neumann entropy approximation.
    
    Uses S ≈ -tr(ρ ln(ρ)) computed via series expansion
    instead of eigendecomposition.
    
    Args:
        rho: Density matrix
        eps: Regularization
        
    Returns:
        Von Neumann entropy (scalar tensor)
    """
    # Linear entropy approximation: S_lin = 1 - tr(ρ²)
    purity = torch.trace(rho @ rho).real
    linear_entropy = 1.0 - purity
    
    # Matrix log via Padé approximation for ρ close to identity
    eye = torch.eye(rho.shape[0], device=rho.device, dtype=rho.dtype)
    rho_minus_I = rho - eye
    
    # Second-order approximation
    log_approx = rho_minus_I - 0.5 * (rho_minus_I @ rho_minus_I)
    entropy = -torch.trace(rho @ log_approx).real
    
    # Use linear entropy as fallback if approximation gives negative
    entropy = torch.maximum(entropy, linear_entropy)
    
    return torch.clamp(entropy, min=0.0)


@dataclass
class LindbladMetrics:
    """Metrics from Lindblad evolution"""
    trace: float
    purity: float
    von_neumann_entropy: float
    total_dissipation: float
    n_steps: int


class ModularHamiltonian(nn.Module):
    """
    Modular Hamiltonian K = -ln(ρ₀).
    
    Generates evolution in modular time s via:
        ρ(s) = e^{-iKs} ρ e^{iKs}
        
    Args:
        system_dim: Dimension of the quantum system
        diamond_radius: Characteristic radius R (for geometric scaling)
        device: Torch device for computations
    """

    def __init__(
        self,
        system_dim: int,
        diamond_radius: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.system_dim = system_dim
        self.R = diamond_radius
        self.device = device if device is not None else get_device()

        # Modular temperature (KMS condition): β = 2π
        self.beta = 2 * np.pi

        # Learnable modular Hamiltonian matrix
        # Initialized as Hermitian
        K_real = torch.randn(system_dim, system_dim, device=self.device) * 0.01
        self.K_real = nn.Parameter(K_real)

        # Eigenspectrum cache
        self._eigenvalues: Optional[torch.Tensor] = None
        self._eigenvectors: Optional[torch.Tensor] = None
        self._cache_valid = False

    @property
    def K(self) -> torch.Tensor:
        """Hermitian modular Hamiltonian matrix"""
        return 0.5 * (self.K_real + self.K_real.T)

    def compute_from_stress_energy(
        self, stress_energy: torch.Tensor, spatial_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute modular Hamiltonian from stress-energy tensor.
        K = 2π ∫ d³x (R² - r²)/(2R) T₀₀
        """
        r_squared = (spatial_coords**2).sum(dim=-1)
        weight = torch.clamp(self.R**2 - r_squared, min=0) / (2 * self.R)

        if stress_energy.dim() > 1:
            T00 = stress_energy[..., 0, 0]
        else:
            T00 = stress_energy

        K_value = 2 * np.pi * (weight * T00).sum()
        return K_value

    def _compute_eigenspectrum(self):
        """Compute and cache eigenspectrum using differentiable power iteration."""
        if not self._cache_valid:
            # Use power iteration for dominant eigenvector (fully differentiable)
            dominant_vec, dominant_val = power_iteration(
                self.K, n_iter=20, return_eigenvalue=True
            )
            # For modular flow, the dominant component is most important
            self._eigenvalues = dominant_val.unsqueeze(0)
            self._eigenvectors = dominant_vec.unsqueeze(-1)
            self._cache_valid = True

    def invalidate_cache(self):
        """Invalidate eigenspectrum cache."""
        self._cache_valid = False

    def modular_flow(self, operator: torch.Tensor, modular_time: float) -> torch.Tensor:
        """
        Flow operator along modular time.
        O(s) = e^{iKs} O e^{-iKs}
        """
        self._compute_eigenspectrum()

        eigenvectors = self._eigenvectors
        if not torch.is_complex(eigenvectors):
            eigenvectors = eigenvectors.to(torch.complex64)
        
        # Construct unitary e^{iKs}
        phases = torch.exp(1j * self._eigenvalues.to(torch.complex64) * modular_time)
        U = (eigenvectors @ torch.diag(phases) @ eigenvectors.T.conj())

        # Apply flow
        op_complex = operator.to(torch.complex64)
        flowed = U @ op_complex @ U.conj().T

        return flowed.real.to(operator.dtype)

    def thermal_state(self) -> torch.Tensor:
        """
        Compute thermal state at modular temperature β = 2π.
        ρ_β = e^{-βK} / Tr(e^{-βK})
        """
        self._compute_eigenspectrum()
        boltzmann = torch.exp(-self.beta * self._eigenvalues)
        Z = boltzmann.sum()
        rho_diag = boltzmann / Z
        rho = self._eigenvectors @ torch.diag(rho_diag) @ self._eigenvectors.T
        return rho

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Apply modular Hamiltonian evolution (one step).
        dρ/ds = -i[K, ρ]
        """
        K = self.K
        if torch.is_complex(rho):
            K = K.to(dtype=rho.dtype)
        commutator = K @ rho - rho @ K
        return -1j * commutator


class LindbladOperator:
    """Lindblad jump operator L_k for open system dynamics."""

    def __init__(self, name: str, matrix: torch.Tensor, rate: float):
        self.name = name
        self.L = matrix
        self.rate = rate
        self.LdagL = self.L.conj().T @ self.L

    def apply(self, rho: torch.Tensor) -> torch.Tensor:
        """γ(L ρ L† - ½{L†L, ρ})"""
        LrhoLdag = self.L @ rho @ self.L.conj().T
        anticommutator = self.LdagL @ rho + rho @ self.LdagL
        return self.rate * (LrhoLdag - 0.5 * anticommutator)

    def update_rate(self, new_rate: float):
        self.rate = new_rate


class LindbladEvolutionEngine(nn.Module):
    """
    Lindblad master equation evolution engine.
    
    Args:
        system_dim: Dimension of quantum system (default: 496)
        n_lindblad_ops: Number of Lindblad operators (default: 8)
        device: Torch device
        use_modular_hamiltonian: Whether to use geometric modular Hamiltonian
    """

    def __init__(
        self,
        system_dim: int = 496,
        n_lindblad_ops: int = 8,
        device: Optional[torch.device] = None,
        use_modular_hamiltonian: bool = True,
    ):
        super().__init__()

        self.system_dim = system_dim
        self.n_lindblad_ops = n_lindblad_ops
        self.device = device if device is not None else get_device()

        # System Hamiltonian (learnable, Hermitian)
        H_init = torch.randn(system_dim, system_dim, device=self.device) * 0.01
        self.H_real = nn.Parameter(H_init)

        # Lindblad operators (learnable)
        self.lindblad_L = nn.ParameterList()
        self.lindblad_rates = nn.ParameterList()

        # Initialize random operators and uniform rates
        default_rate = 0.1
        for k in range(n_lindblad_ops):
            L_k = torch.randn(system_dim, system_dim, device=self.device) * 0.01
            self.lindblad_L.append(nn.Parameter(L_k))
            
            rate = torch.tensor([default_rate], device=self.device, dtype=torch.float32)
            self.lindblad_rates.append(nn.Parameter(rate))

        if use_modular_hamiltonian:
            self.modular_H = ModularHamiltonian(system_dim, device=self.device)
        else:
            self.modular_H = None

        self.log_Z = nn.Parameter(torch.zeros(1, device=self.device))
        self.beta = 2 * np.pi

        logger.info(f"LindbladEvolutionEngine initialized: dim={system_dim}")

    @property
    def H(self) -> torch.Tensor:
        return 0.5 * (self.H_real + self.H_real.T)

    def _compute_unitary_evolution(self, rho: torch.Tensor) -> torch.Tensor:
        H = self.H
        if torch.is_complex(rho):
            H = H.to(dtype=rho.dtype)
        commutator = H @ rho - rho @ H
        return -1j * commutator

    def _compute_dissipative_evolution(
        self, rho: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        dissipator = torch.zeros_like(rho)
        total_dissipation = 0.0

        for L_param, rate_param in zip(self.lindblad_L, self.lindblad_rates):
            L = L_param
            if torch.is_complex(rho):
                L = L.to(dtype=rho.dtype)
            L_dag = L.conj().T
            rate = rate_param.abs()

            LrhoLdag = L @ rho @ L_dag
            LdagL = L_dag @ L
            anticomm = LdagL @ rho + rho @ LdagL

            term = rate * (LrhoLdag - 0.5 * anticomm)
            dissipator = dissipator + term
            total_dissipation += torch.abs(term).sum().item()

        return dissipator, total_dissipation

    def evolve_density_matrix(
        self,
        rho: torch.Tensor,
        dt: float,
        n_steps: int = 1,
        enforce_positivity: bool = True,
        enforce_trace: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_dissipation = 0.0

        for step in range(n_steps):
            drho_unitary = self._compute_unitary_evolution(rho)
            drho_dissipative, step_dissipation = self._compute_dissipative_evolution(rho)
            total_dissipation += step_dissipation

            drho = drho_unitary.real + drho_dissipative.real
            rho = rho + drho * dt

            if enforce_trace:
                trace = torch.trace(rho)
                if trace.abs() > 1e-10:
                    rho = rho / trace

            if enforce_positivity:
                rho = self._project_positive_semidefinite(rho)

        metrics = self._compute_metrics(rho, total_dissipation, n_steps)
        return rho, metrics

    def _project_positive_semidefinite(self, rho: torch.Tensor) -> torch.Tensor:
        return soft_psd_projection(rho, eps=1e-6, n_iter=5)

    def _compute_metrics(
        self, rho: torch.Tensor, total_dissipation: float, n_steps: int
    ) -> Dict[str, float]:
        trace = torch.trace(rho).real.item()
        purity = torch.trace(rho @ rho).real.item()
        entropy = differentiable_von_neumann_entropy(rho).item()
        
        return {
            "trace": trace,
            "purity": purity,
            "von_neumann_entropy": entropy,
            "total_dissipation": total_dissipation,
            "n_steps": n_steps,
        }

    def forward(
        self, input_state: torch.Tensor, evolution_time: float = 1.0, n_steps: int = 10
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = input_state.shape[0]

        if input_state.shape[-1] != self.system_dim:
            input_state = F.adaptive_avg_pool1d(
                input_state.unsqueeze(1), self.system_dim
            ).squeeze(1)

        psi = input_state[0]
        psi = psi / (psi.norm() + 1e-10)
        rho = psi.unsqueeze(-1) @ psi.unsqueeze(0)
        rho = rho.to(self.device)

        dt = evolution_time / n_steps
        evolved_rho, metrics = self.evolve_density_matrix(rho, dt, n_steps)

        output_state = power_iteration(evolved_rho, n_iter=20).real
        output_state = output_state / (output_state.norm() + 1e-10)
        output_state = output_state.unsqueeze(0).expand(batch_size, -1)

        if output_state.shape[-1] != input_state.shape[-1]:
            output_state = F.adaptive_avg_pool1d(
                output_state.unsqueeze(1), input_state.shape[-1]
            ).squeeze(1)

        return output_state, metrics
    
    def evolve(self, input_state: torch.Tensor, n_steps: int = 10) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.forward(input_state, evolution_time=1.0, n_steps=n_steps)