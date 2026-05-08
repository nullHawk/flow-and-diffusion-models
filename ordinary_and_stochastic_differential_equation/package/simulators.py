from . import Simulator, ODE, SDE
import torch
import math

class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        xt_h = xt + h * self.ode.drift_coefficient(xt, t)
        return xt_h

class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        xt_h = xt + h * self.sde.drift_coefficient(xt, t) + math.sqrt(h) * self.sde.diffusion_coefficient(xt, t)
        return xt_h