# DUFFING OSCILLATOR
def DuffingOscillatorDynamics(x, y, alpha, beta, delta):

    timestep = 0.25
    xdot = y
    ydot = -delta*y - x*(alpha*(x**2) + beta)

    return [x + timestep*xdot, y + timestep*ydot]