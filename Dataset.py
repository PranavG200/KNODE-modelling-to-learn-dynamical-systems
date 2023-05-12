<<<<<<< HEAD
import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint
from Dynamics import DuffingOscillator, DoubleGyre, BickleyJet

def GenerateData(GridSize, flag):

    # Creating parametric space to take into account variations in parameters
    if flag == 2:
        XG = np.linspace(0,20,GridSize)
        YG = np.linspace(-3,3,GridSize)

    elif flag == 1:
        XG = np.linspace(0,2,GridSize)
        YG = np.linspace(0,1,GridSize)  

    elif flag == 0:

        XG = np.linspace(-2,2,GridSize)
        YG = np.linspace(0.1,1,GridSize)

    TrainSize = GridSize**2

    XTrain = np.array(np.meshgrid(XG, YG))
    xTrain = np.hstack((XTrain[0].reshape(TrainSize,1), XTrain[1].reshape(TrainSize,1)))

    return xTrain, TrainSize

def GenerateTrainData(TrainSize, Size, IC, t, Model):

    # Create Training data by solving ODE and defining actual trajectories for the given initial conditions
    
    yTrain = torch.empty(TrainSize, Size, 2)
    with torch.no_grad():
        for i in range(TrainSize):
            y0 = torch.tensor([IC[i,0], IC[i,1]])
            yTrain[i] = odeint(Model, y0, t, method='dopri5')

    yTrain = yTrain.numpy()

=======
import numpy as np
import torch
from torchdiffeq import odeint_adjoint as odeint
from Dynamics import DuffingOscillator, DoubleGyre, BickleyJet

def GenerateData(GridSize, flag):

    # Creating parametric space to take into account variations in parameters
    if flag == 2:
        XG = np.linspace(0,20,GridSize)
        YG = np.linspace(-3,3,GridSize)

    elif flag == 1:
        XG = np.linspace(0,2,GridSize)
        YG = np.linspace(0,1,GridSize)  

    elif flag == 0:

        XG = np.linspace(-2,2,GridSize)
        YG = np.linspace(0.1,1,GridSize)

    TrainSize = GridSize**2

    XTrain = np.array(np.meshgrid(XG, YG))
    xTrain = np.hstack((XTrain[0].reshape(TrainSize,1), XTrain[1].reshape(TrainSize,1)))

    return xTrain, TrainSize

def GenerateTrainData(TrainSize, Size, IC, t, Model):

    # Create Training data by solving ODE and defining actual trajectories for the given initial conditions
    
    yTrain = torch.empty(TrainSize, Size, 2)
    with torch.no_grad():
        for i in range(TrainSize):
            y0 = torch.tensor([IC[i,0], IC[i,1]])
            yTrain[i] = odeint(Model, y0, t, method='dopri5')

    yTrain = yTrain.numpy()

>>>>>>> 98cab118871594355766d015a467f566878df94d
    return yTrain