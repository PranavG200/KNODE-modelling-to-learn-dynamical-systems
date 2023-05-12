import numpy as np
from sympy import Symbol, lambdify
import sympy as sp
import torch
import torch.nn as nn

# DUFFING OSCILLATOR

class DuffingOscillator(nn.Module):

    def __init__(self):
        super().__init__()
        self.Alpha = torch.nn.Parameter(-1*torch.rand(1) + 1.5)
        self.Beta = torch.nn.Parameter(-1*torch.rand(1) - 0.5)
        self.Delta = torch.nn.Parameter(-1*torch.rand(1) + 1)

    def DuffingOscillatorDynamics(self, y, t, alpha, beta, delta):

        xdot = y[1]
        ydot = -delta*y[1] - y[0]*(alpha*(y[0]**2) + beta)

        return [xdot, ydot]

    def forward(self, t, y):
        return torch.tensor(self.DuffingOscillatorDynamics(y.T, t, self.Alpha, self.Beta, self.Delta))

# DOUBLE GYRE MODEL

class DoubleGyre(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = torch.nn.Parameter(-0.1*torch.rand(1) + 0.3)
        self.alpha = torch.nn.Parameter(-0.1*torch.rand(1) + 0.3)
        self.A = torch.nn.Parameter(-0.1*torch.rand(1) + 0.3)
        self.omega = torch.nn.Parameter(-0.2*np.pi*torch.rand(1) + 2.1*np.pi)
    
    def DoubleGyreDynamics(self, y, t, eps, alpha, A, omega):

        func = eps*np.sin(omega*t)* (y[0]**2) + (1-2*eps*np.sin(omega*t))*y[0]
        Dfunc = 2*alpha*np.sin(omega*t)*y[0] + 1-2*alpha*np.sin(omega*t)

        xdot = -np.pi*A*np.sin(np.pi*func)*np.cos(np.pi*y[1])
        ydot = np.pi*A*np.cos(np.pi*func)*np.sin(np.pi*y[1])*Dfunc
        
        return [xdot, ydot]

    def forward(self, t, y):
        return torch.tensor(self.DoubleGyreDynamics(y.T, t, self.eps, self.alpha, self.A, self.omega))

# BICKLEY JET MODEL

class BickleyJet(nn.Module):

    def __init__(self):
        super().__init__()
        self.U0 = torch.nn.Parameter(-0.13*torch.rand(1) + 5.33)
        self.L0 = torch.nn.Parameter(-0.13*torch.rand(1) + 1.85)
        self.eps1 = torch.nn.Parameter(-0.01*torch.rand(1) + 0.08)
        self.eps2 = torch.nn.Parameter(-0.08*torch.rand(1) + 0.44)
        self.eps3 = torch.nn.Parameter(-0.08*torch.rand(1) + 0.34)
        self.R0 = torch.nn.Parameter(-0.14*torch.rand(1) + 6.42)
        self.Dphix, self.Dphiy = self.BickleyJetDynamicsDerivs()

    def BickleyJetDynamics(self, y, t, U0, L0, eps1, eps2, eps3, R0, Dphix, Dphiy):

        xdot = Dphix(y[0], y[1], t, U0, L0, eps1, eps2, eps3, R0)
        ydot = Dphiy(y[0], y[1], t, U0, L0, eps1, eps2, eps3, R0)

        return [xdot, ydot]

    def BickleyJetDynamicsDerivs(self): 

        X = Symbol('x', real=True)
        Y = Symbol('y', real=True)
        U0 = Symbol('U0', real=True)
        L0 = Symbol('L0', real=True)
        eps1 = Symbol('eps1', real=True)
        eps2 = Symbol('eps2', real=True)
        eps3 = Symbol('eps3', real=True)
        R0 = Symbol('R0', real=True)
        t = Symbol('t', real=True)

        C1, C2, C3 = 0.1446*U0, 0.2053*U0, 0.4561*U0
        K1, K2, K3 = 2/R0, 4/R0, 6/R0
        Phi0 = -U0*L0*sp.tanh(Y/L0)

        f1 = eps1*sp.exp(-sp.im(K1*C1*t)) 
        f2 = eps2*sp.exp(-sp.im(K2*C2*t))
        f3 = eps3*sp.exp(-sp.im(K3*C3*t))

        Phi1 = U0*L0*(1/sp.cosh(Y/L0)**2) + (f1*sp.exp(sp.im(K1*X)) + f2*sp.exp(sp.im(K2*X)) + f3*sp.exp(sp.im(K3*X)))
        Phi = Phi0 + Phi1

        Dxprime = Phi.diff(X)
        Dyprime = Phi.diff(Y)

        Dphix = lambdify([X, Y, t, U0, L0, eps1, eps2, eps3, R0], Dxprime, 'numpy')
        Dphiy = lambdify([X, Y, t, U0, L0, eps1, eps2, eps3, R0], Dyprime, 'numpy')

        return Dphix, Dphiy

    def forward(self, t, y): 
        return torch.tensor(self.BickleyJetDynamics(y.T, t, self.U0, self.L0, self.eps1, self.eps2, self.eps3, self.R0, self.Dphix, self.Dphiy))
