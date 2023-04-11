# BICKLEY JET MODEL

from sympy import Symbol, lambdify
import sympy as sp

def BickleyJetDynamicsDerivs():
    
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

def BickleyJetDynamics(x, y, t, U0, L0, eps1, eps2, eps3, R0, Dphix, Dphiy):

    timestep = 0.1
    xdot = Dphix(x, y, t, U0, L0, eps1, eps2, eps3, R0)
    ydot = Dphiy(x, y, t, U0, L0, eps1, eps2, eps3, R0)

    return [x + timestep*xdot, y + timestep*ydot]
