# DOUBLE GYRE MODEL
def DoubleGyreDynamics(x, y, t, eps, alpha, A, omega):

    func = eps*np.sin(omega*t)* (x**2) + (1-2*eps*np.sin(omega*t))*x
    Dfunc = 2*alpha*np.sin(omega*t)*x + 1-2*alpha*np.sin(omega*t)

    timestep = 0.1
    xdot = -np.pi*A*np.sin(np.pi*func)*np.cos(np.pi*y)
    ydot = np.pi*A*np.cos(np.pi*func)*np.sin(np.pi*y)*Dfunc
    
    return [x + (timestep)*xdot, y + (timestep)*ydot]