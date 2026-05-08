from . import ODE, SDE
import torch

class BrownianMotion(SDE):
    def __init__(self, sigma: float):
        self.sigma = sigma
        
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape ()
        Returns:
            - drift: shape (bs, dim)
        """
        ut = torch.zeros_like(xt)
        return ut

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """`
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape ()
        Returns:
            - diffusion: shape (bs, dim)
        """
        e = torch.randn_like(xt)
        return self.sigma * torch.ones_like(xt) * e
        
