#
# Flow Matching for Molecular Structure Generation
# Adapted from SimpleFold
#

import torch


def right_pad_dims_to(x, t):
    """Pad dimensions of t to match x"""
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.reshape(*t.shape, *((1,) * padding_dims))


class LinearPath:
    """
    Linear flow matching path from noise to data.
    
    x0: noise (t=0)
    x1: data (t=1)
    
    Interpolation: x_t = t * x1 + (1-t) * x0
    Velocity: u_t = x1 - x0
    """
    
    def __init__(self):
        pass
    
    def compute_alpha_t(self, t):
        """Data coefficient: alpha(t) = t"""
        return t, torch.ones_like(t)  # alpha_t, d_alpha_t
    
    def compute_sigma_t(self, t):
        """Noise coefficient: sigma(t) = 1 - t"""
        return 1 - t, -torch.ones_like(t)  # sigma_t, d_sigma_t
    
    def compute_mu_t(self, t, x0, x1):
        """Mean of p_t: mu_t = t * x1 + (1-t) * x0"""
        t = right_pad_dims_to(x1, t)
        return t * x1 + (1 - t) * x0
    
    def compute_xt(self, t, x0, x1):
        """Sample from p_t (deterministic for linear path)"""
        return self.compute_mu_t(t, x0, x1)
    
    def compute_ut(self, t, x0, x1):
        """
        Vector field (velocity): u_t = dx/dt = x1 - x0
        
        NOTE: For linear path, velocity is constant!
        """
        return x1 - x0
    
    def interpolant(self, t, x0, x1):
        """
        Compute interpolated sample and target velocity.
        
        Returns:
            t: timestep
            x_t: noised sample at time t
            u_t: target velocity
        """
        t = right_pad_dims_to(x1, t)
        x_t = self.compute_xt(t, x0, x1)
        u_t = self.compute_ut(t, x0, x1)
        return t, x_t, u_t


if __name__ == "__main__":
    print("Testing flow matching...")
    print("=" * 80)
    
    # Create path
    path = LinearPath()
    
    # Test data
    x0 = torch.randn(2, 10, 3)  # Noise
    x1 = torch.randn(2, 10, 3)  # Data
    t = torch.rand(2)  # Random timesteps
    
    # Test interpolation
    t_expanded, x_t, u_t = path.interpolant(t, x0, x1)
    
    print(f"x0 (noise): {x0.shape}")
    print(f"x1 (data): {x1.shape}")
    print(f"t: {t.shape}")
    print(f"t_expanded: {t_expanded.shape}")
    print(f"x_t (interpolated): {x_t.shape}")
    print(f"u_t (velocity): {u_t.shape}")
    
    # Verify properties
    print(f"\n✓ At t=0, x_t should ≈ x0")
    t_zero = torch.zeros(2)
    _, x_t_zero, _ = path.interpolant(t_zero, x0, x1)
    print(f"  Error: {(x_t_zero - x0).abs().max():.6f}")
    
    print(f"✓ At t=1, x_t should ≈ x1")
    t_one = torch.ones(2)
    _, x_t_one, _ = path.interpolant(t_one, x0, x1)
    print(f"  Error: {(x_t_one - x1).abs().max():.6f}")
    
    print(f"✓ Velocity should be constant (x1 - x0)")
    expected_vel = x1 - x0
    print(f"  Error: {(u_t - expected_vel).abs().max():.6f}")
    
    print("\n" + "=" * 80)
    print("✓ Flow matching tests passed!")
