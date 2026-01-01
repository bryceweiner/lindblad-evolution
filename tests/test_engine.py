import torch
import pytest
from lindblad_evolution import LindbladEvolutionEngine

def test_initialization():
    engine = LindbladEvolutionEngine(system_dim=16)
    assert engine.system_dim == 16

def test_forward_pass_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    
    device = torch.device("mps")
    engine = LindbladEvolutionEngine(system_dim=16, device=device)
    input_state = torch.randn(1, 16, device=device)
    
    # Run forward pass
    output, metrics = engine(input_state)
    
    assert output.shape == (1, 16)
    assert metrics["purity"] <= 1.0
    assert metrics["trace"] > 0.99 # Should be close to 1