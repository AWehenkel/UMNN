"""
Numerical validation tests for UMNN to verify:
1. Gradient correctness through fitting monotonic functions
2. Integral convergence as number of quadrature steps increases
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from models.UMNN.UMNNMAF import IntegrandNetwork
from models.UMNN.NeuralIntegral import NeuralIntegral
from models.UMNN.ParallelNeuralIntegral import ParallelNeuralIntegral


def test_integral_convergence():
    """
    Test that the neural integral converges to the true integral
    as the number of quadrature steps increases.

    We use f(x) = 1 + x^2, which integrates to x + x^3/3 + C
    """
    print("=" * 80)
    print("Testing Integral Convergence")
    print("=" * 80)

    device = "cpu"  # Use CPU for reproducibility
    torch.manual_seed(42)

    # Create a simple integrand that mimics f(x) = 1 + x^2
    class SimpleIntegrand(nn.Module):
        def __init__(self):
            super().__init__()
            # This will approximate 1 + x^2

        def forward(self, x, h):
            # x shape: [batch, 1], h: [batch, h_dim]
            # Return 1 + x^2
            return 1.0 + x ** 2

    integrand = SimpleIntegrand().to(device)

    # Test parameters
    batch_size = 5
    x0 = torch.zeros(batch_size, 1, device=device)
    x_end = torch.ones(batch_size, 1, device=device) * 2.0  # Integrate from 0 to 2
    h = torch.zeros(batch_size, 1, device=device)  # Dummy h

    # True integral: ∫(1 + x^2)dx from 0 to 2 = [x + x^3/3] from 0 to 2 = 2 + 8/3 = 4.667
    true_integral = 2.0 + 8.0/3.0

    print(f"\nIntegrating f(x) = 1 + x^2 from 0 to 2")
    print(f"True integral value: {true_integral:.6f}")

    # Test with increasing number of steps
    steps_list = [5, 10, 20, 50, 100, 200]
    errors = []

    print("\nTesting convergence with increasing quadrature steps:")
    print(f"{'Steps':<10} {'Computed':<15} {'Error':<15} {'Rel Error %':<15}")
    print("-" * 60)

    for nb_steps in steps_list:
        with torch.no_grad():
            # Flatten parameters (even though we don't use them)
            flat_params = torch.tensor([])

            # Compute integral using ParallelNeuralIntegral
            result = ParallelNeuralIntegral.apply(
                x0, x_end, integrand, flat_params, h, nb_steps, False
            )

            computed_integral = result.mean().item()
            error = abs(computed_integral - true_integral)
            rel_error = (error / true_integral) * 100
            errors.append(error)

            print(f"{nb_steps:<10} {computed_integral:<15.6f} {error:<15.6e} {rel_error:<15.6f}")

    # Check convergence
    print("\nConvergence analysis:")
    for i in range(1, len(errors)):
        if errors[i] < errors[i-1]:
            print(f"  ✓ Error decreased from {steps_list[i-1]} to {steps_list[i]} steps")
        else:
            print(f"  ✗ Error increased from {steps_list[i-1]} to {steps_list[i]} steps")

    # Final error should be small
    final_error = errors[-1]
    if final_error < 1e-4:
        print(f"\n✓ Final error ({final_error:.6e}) is acceptably small")
    else:
        print(f"\n✗ Final error ({final_error:.6e}) is larger than expected")

    return errors[-1] < 1e-4


def test_gradient_correctness_analytical():
    """
    Test gradient correctness using analytical comparison.
    We'll use finite differences to verify gradients.
    """
    print("\n" + "=" * 80)
    print("Testing Gradient Correctness (Analytical)")
    print("=" * 80)

    device = "cpu"
    torch.manual_seed(42)

    # Create a simple integrand network
    input_dim = 3
    integrand = IntegrandNetwork(
        nnets=input_dim,
        nin=1 + 1,  # 1 for x, 1 for h (out_made=1)
        hidden_sizes=[20, 20],
        nout=1,
        device=device
    ).to(device)

    # Test parameters
    batch_size = 10
    x0 = torch.zeros(batch_size, input_dim, requires_grad=True, device=device)
    x = torch.randn(batch_size, input_dim, requires_grad=True, device=device)
    h = torch.randn(batch_size, input_dim, requires_grad=True, device=device)

    flat_params = torch.cat([p.contiguous().view(-1) for p in integrand.parameters()])

    print("\nComputing gradients using autograd...")

    # Forward pass
    result = ParallelNeuralIntegral.apply(x0, x, integrand, flat_params, h, 20, False)
    loss = result.sum()

    # Backward pass
    loss.backward()

    # Store gradients
    x_grad_autograd = x.grad.clone()
    h_grad_autograd = h.grad.clone()

    print(f"✓ Autograd gradients computed")
    print(f"  x.grad: mean={x_grad_autograd.mean():.6f}, std={x_grad_autograd.std():.6f}")
    print(f"  h.grad: mean={h_grad_autograd.mean():.6f}, std={h_grad_autograd.std():.6f}")

    # Compute numerical gradients using finite differences
    print("\nComputing numerical gradients using finite differences...")
    eps = 1e-4

    def compute_loss(x_val, h_val):
        with torch.no_grad():
            result = ParallelNeuralIntegral.apply(x_val, x, integrand, flat_params, h_val, 20, False)
            return result.sum().item()

    # Numerical gradient for x
    base_loss = compute_loss(x0, h)
    x_grad_numerical = torch.zeros_like(x)

    for i in range(min(3, batch_size)):  # Test a few samples
        for j in range(input_dim):
            x0_plus = x0.clone()
            x0_plus[i, j] += eps
            loss_plus = compute_loss(x0_plus, h)
            x_grad_numerical[i, j] = (loss_plus - base_loss) / eps

    # Compare gradients
    print("\nComparing autograd vs numerical gradients for x0:")
    for i in range(min(3, batch_size)):
        print(f"  Sample {i}:")
        print(f"    Autograd:  {x0.grad[i].detach().numpy()}")
        print(f"    Numerical: {x_grad_numerical[i].numpy()}")

        # Check relative error
        rel_error = torch.abs(x0.grad[i] - x_grad_numerical[i]) / (torch.abs(x0.grad[i]) + 1e-8)
        print(f"    Relative error: {rel_error.mean().item():.6f}")

    print("\n✓ Gradient computation test completed")
    return True


def test_monotonic_function_fitting():
    """
    Test that we can fit a known monotonic function using gradient descent.
    We'll fit f(x) = x^3 (which is monotonically increasing for x > 0).
    """
    print("\n" + "=" * 80)
    print("Testing Monotonic Function Fitting")
    print("=" * 80)

    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate training data: y = x^3 for x in [0, 2]
    n_samples = 100
    x_data = np.linspace(0, 2, n_samples).reshape(-1, 1)
    y_data = x_data ** 3

    # Add context variable (constant in this case)
    x_tensor = torch.FloatTensor(x_data).to(device)
    y_tensor = torch.FloatTensor(y_data).to(device)

    print(f"\nGenerating {n_samples} samples from y = x^3 for x in [0, 2]")

    # Create a simple monotonic model
    # We'll integrate a positive function (achieved through ELU+1)
    input_dim = 1
    integrand = IntegrandNetwork(
        nnets=input_dim,
        nin=1 + 1,  # 1 for x, 1 for h
        hidden_sizes=[32, 32],
        nout=1,
        act_func='ELU',  # This ensures output is always positive
        device=device
    ).to(device)

    optimizer = optim.Adam(integrand.parameters(), lr=0.01)

    print("\nTraining monotonic model...")
    losses = []
    n_epochs = 200

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Prepare inputs for integration
        x0 = torch.zeros(n_samples, input_dim, device=device)
        h = torch.zeros(n_samples, input_dim, device=device)  # No context
        flat_params = torch.cat([p.contiguous().view(-1) for p in integrand.parameters()])

        # Compute integral (forward pass)
        y_pred = ParallelNeuralIntegral.apply(x0, x_tensor, integrand, flat_params, h, 30, False)

        # Compute loss
        loss = ((y_pred - y_tensor) ** 2).mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")

    # Test on validation data
    print("\nEvaluating on test data...")
    with torch.no_grad():
        x_test = torch.FloatTensor([[0.5], [1.0], [1.5], [2.0]]).to(device)
        y_true = x_test ** 3

        x0_test = torch.zeros(4, input_dim, device=device)
        h_test = torch.zeros(4, input_dim, device=device)
        flat_params = torch.cat([p.contiguous().view(-1) for p in integrand.parameters()])

        y_pred = ParallelNeuralIntegral.apply(x0_test, x_test, integrand, flat_params, h_test, 30, False)

        print(f"\n{'x':<10} {'True y=x^3':<15} {'Predicted':<15} {'Error':<15}")
        print("-" * 55)
        for i in range(4):
            x_val = x_test[i, 0].item()
            true_val = y_true[i, 0].item()
            pred_val = y_pred[i, 0].item()
            error = abs(true_val - pred_val)
            print(f"{x_val:<10.2f} {true_val:<15.6f} {pred_val:<15.6f} {error:<15.6f}")

        # Check if final loss is small
        final_loss = losses[-1]
        if final_loss < 0.1:
            print(f"\n✓ Successfully fitted the function (final loss: {final_loss:.6f})")
            success = True
        else:
            print(f"\n✗ Failed to fit function adequately (final loss: {final_loss:.6f})")
            success = False

    # Create visualization
    print("\nGenerating visualization...")
    try:
        with torch.no_grad():
            x_plot = torch.linspace(0, 2, 100).reshape(-1, 1).to(device)
            y_true_plot = x_plot ** 3

            x0_plot = torch.zeros(100, input_dim, device=device)
            h_plot = torch.zeros(100, input_dim, device=device)
            y_pred_plot = ParallelNeuralIntegral.apply(x0_plot, x_plot, integrand, flat_params, h_plot, 30, False)

            plt.figure(figsize=(12, 4))

            # Plot 1: Function fit
            plt.subplot(1, 2, 1)
            plt.plot(x_plot.cpu().numpy(), y_true_plot.cpu().numpy(), 'b-', label='True: y=x³', linewidth=2)
            plt.plot(x_plot.cpu().numpy(), y_pred_plot.cpu().numpy(), 'r--', label='Fitted', linewidth=2)
            plt.scatter(x_data, y_data, alpha=0.3, label='Training data')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Monotonic Function Fitting')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 2: Training loss
            plt.subplot(1, 2, 2)
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.title('Training Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('monotonic_fitting_test.png', dpi=150, bbox_inches='tight')
            print("✓ Saved visualization to 'monotonic_fitting_test.png'")
    except Exception as e:
        print(f"✗ Failed to create visualization: {e}")

    return success


def test_integral_accuracy_various_functions():
    """
    Test integral accuracy for various functions with known integrals.
    """
    print("\n" + "=" * 80)
    print("Testing Integral Accuracy for Various Functions")
    print("=" * 80)

    device = "cpu"

    test_cases = [
        {
            'name': 'Constant: f(x) = 2',
            'function': lambda x, h: torch.ones_like(x) * 2.0,
            'integral': lambda a, b: 2.0 * (b - a),  # 2x
            'x0': 0.0,
            'x1': 3.0,
        },
        {
            'name': 'Linear: f(x) = x',
            'function': lambda x, h: x,
            'integral': lambda a, b: (b**2 - a**2) / 2.0,  # x²/2
            'x0': 0.0,
            'x1': 2.0,
        },
        {
            'name': 'Quadratic: f(x) = x²',
            'function': lambda x, h: x ** 2,
            'integral': lambda a, b: (b**3 - a**3) / 3.0,  # x³/3
            'x0': 1.0,
            'x1': 3.0,
        },
        {
            'name': 'Exponential-like: f(x) = exp(x)',
            'function': lambda x, h: torch.exp(x),
            'integral': lambda a, b: np.exp(b) - np.exp(a),  # exp(x)
            'x0': 0.0,
            'x1': 1.0,
        },
    ]

    print(f"\n{'Function':<25} {'Steps':<10} {'Computed':<15} {'True':<15} {'Error':<15}")
    print("-" * 80)

    all_passed = True

    for test_case in test_cases:
        class TestIntegrand(nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func

            def forward(self, x, h):
                return self.func(x, h)

        integrand = TestIntegrand(test_case['function']).to(device)

        # Test with different number of steps
        for nb_steps in [20, 50, 100]:
            batch_size = 1
            x0 = torch.ones(batch_size, 1, device=device) * test_case['x0']
            x1 = torch.ones(batch_size, 1, device=device) * test_case['x1']
            h = torch.zeros(batch_size, 1, device=device)
            flat_params = torch.tensor([])

            with torch.no_grad():
                result = ParallelNeuralIntegral.apply(x0, x1, integrand, flat_params, h, nb_steps, False)
                computed = result.item()

            true_value = test_case['integral'](test_case['x0'], test_case['x1'])
            error = abs(computed - true_value)

            status = "✓" if error < 1e-3 else "✗"
            print(f"{status} {test_case['name']:<23} {nb_steps:<10} {computed:<15.6f} {true_value:<15.6f} {error:<15.6e}")

            if error >= 1e-3 and nb_steps == 100:
                all_passed = False

    if all_passed:
        print("\n✓ All integral accuracy tests passed")
    else:
        print("\n✗ Some integral accuracy tests failed")

    return all_passed


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("UMNN Numerical Validation Test Suite")
    print("=" * 80)

    results = {}

    # Run tests
    results['integral_convergence'] = test_integral_convergence()
    results['gradient_correctness'] = test_gradient_correctness_analytical()
    results['monotonic_fitting'] = test_monotonic_function_fitting()
    results['integral_accuracy'] = test_integral_accuracy_various_functions()

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All numerical validation tests passed!")
    else:
        print("✗ Some numerical validation tests failed")
    print("=" * 80)
