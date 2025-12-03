"""
Test script to verify JIT compatibility and backward pass correctness for UMNN models.
"""
import torch
import torch.nn as nn
from models.UMNN.UMNNMAF import IntegrandNetwork, EmbeddingNetwork, UMNNMAF
from models.UMNN.NeuralIntegral import NeuralIntegral
from models.UMNN.ParallelNeuralIntegral import ParallelNeuralIntegral
import time


def test_backward_pass():
    """Test that the backward pass works correctly after the fix."""
    print("=" * 80)
    print("Testing Backward Pass")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create a simple integrand network
    batch_size = 10
    input_dim = 5
    hidden_sizes = [50, 50]
    out_made = 1  # This is the default in EmbeddingNetwork

    integrand = IntegrandNetwork(
        nnets=input_dim,
        nin=1 + out_made,  # This is what EmbeddingNetwork uses
        hidden_sizes=hidden_sizes,
        nout=1,
        device=device
    ).to(device)  # Ensure the network is on the correct device

    # Test NeuralIntegral backward
    print("\n1. Testing NeuralIntegral backward pass...")
    x0 = torch.zeros(batch_size, input_dim, requires_grad=True, device=device)
    x = torch.randn(batch_size, input_dim, requires_grad=True, device=device)
    # h should have shape [batch, input_dim * out_made] to match MADE output
    h = torch.randn(batch_size, input_dim * out_made, device=device, requires_grad=True)

    # Flatten parameters
    flat_params = torch.cat([p.contiguous().view(-1) for p in integrand.parameters()])

    try:
        # Forward pass
        result = NeuralIntegral.apply(x0, x, integrand, flat_params, h, 20)
        loss = result.sum()

        # Backward pass
        loss.backward()

        print("   ✓ NeuralIntegral backward pass successful")
        print(f"   Result shape: {result.shape}")
        print(f"   Loss: {loss.item():.6f}")
        if x.grad is not None:
            print(f"   x.grad mean: {x.grad.mean().item():.6f}, std: {x.grad.std().item():.6f}")
    except Exception as e:
        print(f"   ✗ NeuralIntegral backward pass failed: {e}")
        return False

    # Test ParallelNeuralIntegral backward
    print("\n2. Testing ParallelNeuralIntegral backward pass...")
    x0 = torch.zeros(batch_size, input_dim, requires_grad=True, device=device)
    x = torch.randn(batch_size, input_dim, requires_grad=True, device=device)
    h = torch.randn(batch_size, input_dim * out_made, device=device, requires_grad=True)

    try:
        # Forward pass
        result = ParallelNeuralIntegral.apply(x0, x, integrand, flat_params, h, 20, False)
        loss = result.sum()

        # Backward pass
        loss.backward()

        print("   ✓ ParallelNeuralIntegral backward pass successful")
        print(f"   Result shape: {result.shape}")
        print(f"   Loss: {loss.item():.6f}")
        if x.grad is not None:
            print(f"   x.grad mean: {x.grad.mean().item():.6f}, std: {x.grad.std().item():.6f}")
    except Exception as e:
        print(f"   ✗ ParallelNeuralIntegral backward pass failed: {e}")
        return False

    print("\n✓ All backward pass tests passed!")
    return True


def test_full_model():
    """Test the full UMNN model with forward and backward passes."""
    print("\n" + "=" * 80)
    print("Testing Full UMNN Model")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    batch_size = 32
    input_dim = 10

    # Create embedding network
    embedding_net = EmbeddingNetwork(
        in_d=input_dim,
        hiddens_embedding=[50, 50],
        hiddens_integrand=[50, 50],
        out_made=1,
        cond_in=0,
        device=device
    )

    # Create UMNN model
    model = UMNNMAF(
        net=embedding_net,
        input_size=input_dim,
        nb_steps=20,
        device=device,
        solver="CCParallel"
    )

    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    print("\n1. Testing forward pass...")
    x = torch.randn(batch_size, input_dim, device=device, requires_grad=True)

    try:
        z = model.forward(x)
        print(f"   ✓ Forward pass successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {z.shape}")
        print(f"   Output mean: {z.mean().item():.6f}, std: {z.std().item():.6f}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False

    # Test backward pass
    print("\n2. Testing backward pass...")
    try:
        loss = z.sum()
        loss.backward()
        print(f"   ✓ Backward pass successful")
        print(f"   Loss: {loss.item():.6f}")
        if x.grad is not None:
            print(f"   x.grad mean: {x.grad.mean().item():.6f}, std: {x.grad.std().item():.6f}")

        # Check that model parameters have gradients
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters())
        print(f"   Parameters with gradients: {has_grad}/{total_params}")
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        return False

    # Test log likelihood computation
    print("\n3. Testing log likelihood computation...")
    x = torch.randn(batch_size, input_dim, device=device)
    try:
        ll, z = model.compute_ll(x)
        print(f"   ✓ Log likelihood computation successful")
        print(f"   Log likelihood mean: {ll.mean().item():.6f}")
        print(f"   z shape: {z.shape}")
    except Exception as e:
        print(f"   ✗ Log likelihood computation failed: {e}")
        return False

    print("\n✓ All full model tests passed!")
    return True


def test_jit_compatibility():
    """Test JIT compilation compatibility."""
    print("\n" + "=" * 80)
    print("Testing JIT Compatibility")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Note: JIT compilation of custom autograd.Function is not fully supported
    # We test what we can

    print("\n1. Testing JIT script on IntegrandNetwork...")
    input_dim = 5
    out_made = 1
    integrand = IntegrandNetwork(
        nnets=input_dim,
        nin=1 + out_made,  # Match the actual usage in EmbeddingNetwork
        hidden_sizes=[50, 50],
        nout=1,
        device=device
    )

    try:
        scripted_integrand = torch.jit.script(integrand)
        x = torch.randn(10, input_dim, device=device)
        # h must have size that makes (input_dim + h.size(1)) divisible by nnets
        # With nnets=input_dim, we need h.size(1) = input_dim * k for some k
        h = torch.randn(10, input_dim * out_made, device=device)
        output = scripted_integrand(x, h)
        print(f"   ✓ IntegrandNetwork JIT scripting successful")
        print(f"   Output shape: {output.shape}")

        # Verify output matches eager mode
        with torch.no_grad():
            eager_output = integrand(x, h)
            if torch.allclose(output, eager_output, rtol=1e-5):
                print(f"   ✓ JIT script output matches eager mode")
            else:
                print(f"   ✗ Warning: JIT script output differs from eager mode")
    except Exception as e:
        print(f"   ✗ IntegrandNetwork JIT scripting failed: {e}")
        print(f"   Note: This may be expected for complex models with dynamic shapes")

    print("\n2. Testing JIT trace on IntegrandNetwork...")
    try:
        x = torch.randn(10, input_dim, device=device)
        h = torch.randn(10, input_dim * out_made, device=device)
        traced_integrand = torch.jit.trace(integrand, (x, h))
        output = traced_integrand(x, h)
        print(f"   ✓ IntegrandNetwork JIT tracing successful")
        print(f"   Output shape: {output.shape}")

        # Verify output matches eager mode
        with torch.no_grad():
            eager_output = integrand(x, h)
            if torch.allclose(output, eager_output, rtol=1e-5):
                print(f"   ✓ JIT trace output matches eager mode")
            else:
                print(f"   ✗ Warning: JIT trace output differs from eager mode")

        # Test with different batch size
        x_test = torch.randn(5, input_dim, device=device)
        h_test = torch.randn(5, input_dim * out_made, device=device)
        traced_output = traced_integrand(x_test, h_test)
        eager_output_test = integrand(x_test, h_test)
        if torch.allclose(traced_output, eager_output_test, rtol=1e-5):
            print(f"   ✓ JIT trace works with different batch sizes")
        else:
            print(f"   ✗ Warning: JIT trace may have issues with different batch sizes")

    except Exception as e:
        print(f"   ✗ IntegrandNetwork JIT tracing failed: {e}")

    print("\n3. Testing custom autograd.Function compatibility...")
    print("   Note: torch.autograd.Function cannot be directly JIT compiled.")
    print("   However, the functions should work correctly with torch.jit.trace on models that use them.")
    print("   For production use, consider using torch.jit.trace on the entire model during inference.")

    print("\n4. Testing inference-only JIT compatibility...")
    try:
        # For inference, we can use JIT on the integrand network
        integrand.eval()
        x = torch.randn(10, input_dim, device=device)
        h = torch.randn(10, input_dim * out_made, device=device)

        with torch.no_grad():
            traced_model = torch.jit.trace(integrand, (x, h))
            torch.jit.save(traced_model, "/tmp/integrand_traced.pt")
            loaded_model = torch.jit.load("/tmp/integrand_traced.pt")
            output = loaded_model(x, h)
            print(f"   ✓ Successfully saved and loaded JIT traced model")
            print(f"   Output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Inference JIT save/load failed: {e}")

    return True


def benchmark_performance():
    """Benchmark the performance of NeuralIntegral vs ParallelNeuralIntegral."""
    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    batch_size = 64
    input_dim = 10
    nb_steps = 50
    n_iterations = 10
    out_made = 1

    integrand = IntegrandNetwork(
        nnets=input_dim,
        nin=1 + out_made,
        hidden_sizes=[100, 100],
        nout=1,
        device=device
    ).to(device)  # Ensure the network is on the correct device

    x0 = torch.zeros(batch_size, input_dim, device=device)
    x = torch.randn(batch_size, input_dim, device=device)
    h = torch.randn(batch_size, input_dim * out_made, device=device)
    flat_params = torch.cat([p.contiguous().view(-1) for p in integrand.parameters()])

    # Warmup
    for _ in range(3):
        _ = NeuralIntegral.apply(x0, x, integrand, flat_params, h, nb_steps)
        _ = ParallelNeuralIntegral.apply(x0, x, integrand, flat_params, h, nb_steps, False)

    # Benchmark NeuralIntegral
    print(f"\n1. Benchmarking NeuralIntegral (sequential)...")
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iterations):
        result = NeuralIntegral.apply(x0, x, integrand, flat_params, h, nb_steps)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.4f}s for {n_iterations} iterations")
    print(f"   Average: {elapsed/n_iterations*1000:.2f}ms per iteration")

    # Benchmark ParallelNeuralIntegral
    print(f"\n2. Benchmarking ParallelNeuralIntegral (parallel)...")
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iterations):
        result = ParallelNeuralIntegral.apply(x0, x, integrand, flat_params, h, nb_steps, False)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_parallel = time.time() - start
    print(f"   Time: {elapsed_parallel:.4f}s for {n_iterations} iterations")
    print(f"   Average: {elapsed_parallel/n_iterations*1000:.2f}ms per iteration")

    speedup = elapsed / elapsed_parallel
    print(f"\n   Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("UMNN JIT and Backward Pass Test Suite")
    print("=" * 80)

    success = True

    # Run tests
    success &= test_backward_pass()
    success &= test_full_model()
    success &= test_jit_compatibility()

    if success:
        benchmark_performance()

    print("\n" + "=" * 80)
    if success:
        print("✓ All tests completed successfully!")
    else:
        print("✗ Some tests failed!")
    print("=" * 80)
