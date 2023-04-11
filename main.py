import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde

from DuffingOscillator import DuffingOscillatorDynamics
from DoubleGyre import DoubleGyreDynamics
from BickleyJet import BickleyJetDynamics, BickleyJetDynamicsDerivs


# We will calculate dynamics of the system for a trajectory of length TrajLength for each training point in xTrain( initial conditions and parameters)
def PropogateDynamics(TrajLength, xTrain, flag, Dphix=0, Dphiy=0):

    TrainSize = xTrain.shape[0]
    
    # At each timestep, we have input X and Output YT
    X = np.zeros((TrajLength*TrainSize,2))
    YT = np.zeros((TrajLength*TrainSize,2))

    # We know the exact parameters of the system thereby the exact dynamics which we will store in ActualDynamics. 
    # We will also store the dynamics with the uncertain parameters in UncertDynamics
    ActualDynamics = np.zeros((TrajLength*TrainSize,2))
    UncertDynamics = np.zeros((TrajLength*TrainSize,2))

    for i in range(TrainSize):

        X[i*TrajLength,:] = xTrain[i,0:2]

        for j in range(1,TrajLength):

            if flag == 0:
                
                ActualDynamics[i*TrajLength+j-1,:] = DuffingOscillatorDynamics(X[i*TrajLength+j-1,0], X[i*TrajLength+j-1,1], alpha=1, beta=-1, delta=0.5)
                UncertDynamics[i*TrajLength+j-1,:] = DuffingOscillatorDynamics(X[i*TrajLength+j-1,0], X[i*TrajLength+j-1,1], xTrain[i,2], xTrain[i,3], xTrain[i,4])
            
            elif flag == 1:

                ActualDynamics[i*TrajLength+j-1,:] = DoubleGyreDynamics(X[i*TrajLength+j-1,0], X[i*TrajLength+j-1,1], j, eps=0.25, alpha=0.25, A=0.25, omega=2*np.pi)
                UncertDynamics[i*TrajLength+j-1,:] = DoubleGyreDynamics(X[i*TrajLength+j-1,0], X[i*TrajLength+j-1,1], j, xTrain[i,2], xTrain[i,3], xTrain[i,4], xTrain[i,5])

            else:

                ActualDynamics[i*TrajLength+j-1,:] = BickleyJetDynamics(X[i*TrajLength+j-1,0], X[i*TrajLength+j-1,1], 2*j, U0= 5.4138, L0 = 1.77, eps1 = 0.075, eps2 = 0.4, eps3 = 0.3, R0 = 6.371, Dphix = Dphix, Dphiy = Dphiy)
                UncertDynamics[i*TrajLength+j-1,:] = BickleyJetDynamics(X[i*TrajLength+j-1,0], X[i*TrajLength+j-1,1], 2*j, xTrain[i,2], xTrain[i,3], xTrain[i,4], xTrain[i,5], xTrain[i,6], xTrain[i,7], Dphix, Dphiy)
            
            # since y = f^(x) + nn(x). To train the neural net nn, our input vector is X. but ouptut vector is y - f^(x) + noise ( where y is actual dynamics )
            YT[i*TrajLength+j-1,:] = -UncertDynamics[i*TrajLength + j -1,:] + np.random.normal(0, 0.05, size=(2,)) + ActualDynamics[i*TrajLength+j-1,:]
            X[i*TrajLength+j,:] = ActualDynamics[i*TrajLength+j-1,:]

        if flag == 0:

            ActualDynamics[(i+1)*TrajLength -1,:] = DuffingOscillatorDynamics(X[(i+1)*TrajLength-1,0], X[(i+1)*TrajLength-1,1], alpha=1, beta=-1, delta=0.5)
            UncertDynamics[(i+1)*TrajLength -1,:] = DuffingOscillatorDynamics(X[(i+1)*TrajLength-1,0], X[(i+1)*TrajLength-1,1], xTrain[i,2], xTrain[i,3], xTrain[i,4])
        
        elif flag == 1:
                
            ActualDynamics[(i+1)*TrajLength -1,:] = DoubleGyreDynamics(X[(i+1)*TrajLength-1,0], X[(i+1)*TrajLength-1,1], TrajLength -1, eps=0.25, alpha=0.25, A=0.25, omega=2*np.pi)
            UncertDynamics[(i+1)*TrajLength -1,:] = DoubleGyreDynamics(X[(i+1)*TrajLength-1,0], X[(i+1)*TrajLength-1,1], TrajLength -1, xTrain[i,2], xTrain[i,3], xTrain[i,4], xTrain[i,5])
        
        else:

            ActualDynamics[(i+1)*TrajLength -1,:] = BickleyJetDynamics(X[(i+1)*TrajLength-1,0], X[(i+1)*TrajLength-1,1], TrajLength -1, U0= 5.4138, L0 = 1.77, eps1 = 0.075, eps2 = 0.4, eps3 = 0.3, R0 = 6.371, Dphix = Dphix, Dphiy = Dphiy)
            UncertDynamics[(i+1)*TrajLength -1,:] = BickleyJetDynamics(X[(i+1)*TrajLength-1,0], X[(i+1)*TrajLength-1,1], TrajLength -1, xTrain[i,2], xTrain[i,3], xTrain[i,4], xTrain[i,5], xTrain[i,6], xTrain[i,7], Dphix, Dphiy)

        YT[(i+1)*TrajLength -1,:] = -UncertDynamics[(i+1)*TrajLength-1,:] + np.random.normal(0, 0.05, size=(2,)) + ActualDynamics[(i+1)*TrajLength-1,:] 

    return X, YT, ActualDynamics, UncertDynamics

# Initializing parametric space and solution space
def GenerateData(GridSize, PSpaceSize, flag, Dphix=0, Dphiy=0):

    # Creating parametric space to take into account variations in parameters and create training data using them
    if flag == 2:
        XG = np.linspace(0,20,GridSize)
        YG = np.linspace(-3,3,GridSize)

        U0 = np.linspace(5.35,5.47,PSpaceSize)
        L0 = np.linspace(1.72,1.82,PSpaceSize)
        eps1 = np.linspace(0.07,0.08,int(2*PSpaceSize/3))
        eps2 = np.linspace(0.39,0.41,int(2*PSpaceSize/3))    
        eps3 = np.linspace(0.29,0.31,int(2*PSpaceSize/3))
        R0 = np.linspace(6.3,6.4,PSpaceSize)
        
        # Creating Training data set
        # Since there are 5 parameters, Xtrain will be a (TrainSize,5)
        TrainSize = int(GridSize**2 * (PSpaceSize**6)*(2**3)/(3**3))

        XTrain = np.array(np.meshgrid(XG, YG, U0, L0, eps1, eps2, eps3, R0))
        xTrain = np.hstack((XTrain[0].reshape(TrainSize,1), XTrain[1].reshape(TrainSize,1), XTrain[2].reshape(TrainSize,1), XTrain[3].reshape(TrainSize,1), XTrain[4].reshape(TrainSize,1), XTrain[5].reshape(TrainSize,1), XTrain[6].reshape(TrainSize,1), XTrain[7].reshape(TrainSize,1)))


    elif flag == 1:
        XG = np.linspace(0,2,GridSize)
        YG = np.linspace(0,1,GridSize)

        Eps = np.linspace(0.22,0.3,PSpaceSize)
        Alpha = np.linspace(0.22,0.3,PSpaceSize)
        A = np.linspace(0.22,0.3,PSpaceSize)
        Omega = np.linspace(1.95*2*np.pi,2.05*2*np.pi,PSpaceSize)    

        TrainSize = GridSize**2 * PSpaceSize**4

        XTrain = np.array(np.meshgrid(XG, YG, Eps, Alpha, A, Omega))
        xTrain = np.hstack((XTrain[0].reshape(TrainSize,1), XTrain[1].reshape(TrainSize,1), XTrain[2].reshape(TrainSize,1), XTrain[3].reshape(TrainSize,1), XTrain[4].reshape(TrainSize,1), XTrain[5].reshape(TrainSize,1)))

    elif flag == 0:

        XG = np.linspace(-2,2,GridSize)
        YG = np.linspace(0,1,GridSize)

        Alpha = np.linspace(0.5,1.5,PSpaceSize)
        Beta = np.linspace(-1.5,-0.5,PSpaceSize)
        Delta = np.linspace(0,1,PSpaceSize)  

        TrainSize = GridSize**2 * PSpaceSize**3  

        XTrain = np.array(np.meshgrid(XG, YG, Alpha, Beta, Delta))
        xTrain = np.hstack((XTrain[0].reshape(TrainSize,1), XTrain[1].reshape(TrainSize,1), XTrain[2].reshape(TrainSize,1), XTrain[3].reshape(TrainSize,1), XTrain[4].reshape(TrainSize,1)))

    TrajLength = 15
    X, YT, _ , _= PropogateDynamics(TrajLength, xTrain, flag, Dphix, Dphiy)

    ValTrainSize = int(0.7*TrainSize*TrajLength)
    ValTesTrajLength = ValTrainSize + int(0.2*TrainSize*TrajLength)

    data = dde.data.DataSet(X_train = X[:ValTrainSize,0:2], y_train =YT[:ValTrainSize,:], X_test= X[ValTrainSize:ValTesTrajLength,0:2], y_test=YT[ValTrainSize:ValTesTrajLength,:])

    return data, xTrain

DynamicsModel = { "Duffing": 0, "DoubleGyre":1, "BickleyJet":2 }

# We derive the derivatives required for the Bickley jet dynamics beforehand since they are expensive to compute
Dphix, Dphiy = BickleyJetDynamicsDerivs()

# Define network
layer_size = [2] + [30] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Define a Model
dataD, xTrainD = GenerateData(6, 4, DynamicsModel["Duffing"])
modelDuffing = dde.Model(dataD, net)

layer_size = [2] + [38] * 3 + [2]
net = dde.nn.FNN(layer_size, activation, initializer)
dataG, xTrainG = GenerateData(8, 3, DynamicsModel["DoubleGyre"])
modelDGyre = dde.Model(dataG, net)

layer_size = [2] + [26] * 4 + [2]
net = dde.nn.FNN(layer_size, activation, initializer)
dataB, xTrainB = GenerateData(8, 3, DynamicsModel["BickleyJet"], Dphix, Dphiy)
modelBJet = dde.Model(dataB, net)

# Compile and Train
modelDuffing.compile("adam", lr=0.01) #metrics=["l2 relative error"])
losshistory, train_state = modelDuffing.train(epochs=3000)

modelDGyre.compile("adam", lr=0.01) #metrics=["l2 relative error"])
losshistory, train_state = modelDGyre.train(epochs=3000)

modelBJet.compile("adam", lr=0.01) #metrics=["l2 relative error"])
losshistory, train_state = modelBJet.train(epochs=3000)

# Plot the loss trajectory and solution
#dde.saveplot(losshistory, train_state, issave=True, isplot=True) 

## Calculating trajectory for a random initial point in the training set
XD, YTD, ActualDynamicsD, UncertDynamicsD = PropogateDynamics(15, xTrainD[2100,:].reshape(1,5), DynamicsModel["Duffing"])
UncertDynamicsD = UncertDynamicsD + modelDuffing.predict(XD[:,0:2])

XG, YTG, ActualDynamicsG, UncertDynamicsG = PropogateDynamics(15, xTrainG[2150,:].reshape(1,6), DynamicsModel["DoubleGyre"])
UncertDynamicsG = UncertDynamicsG + modelDGyre.predict(XG[:,0:2])

XB, YTB, ActualDynamicsB, UncertDynamicsB = PropogateDynamics(15, xTrainB[4000,:].reshape(1,8), DynamicsModel["BickleyJet"], Dphix, Dphiy)
UncertDynamicsB = UncertDynamicsB + modelBJet.predict(XB[:,0:2])

figure, axis = plt.subplots(3,1, figsize=(10,10))

axis[0].scatter(ActualDynamicsD[:,0], ActualDynamicsD[:,1])
axis[0].scatter(UncertDynamicsD[:,0], UncertDynamicsD[:,1])
axis[0].set_title("Duffing Oscillator")
axis[0].set_xlim([-2,2])
axis[0].set_ylim([0,1])
axis[0].set_xlabel("x")
axis[0].set_ylabel("y")
axis[0].legend(["Actual Dynamics", "KNODE Dynamics"])

axis[1].scatter(ActualDynamicsG[:,0], ActualDynamicsG[:,1])
axis[1].scatter(UncertDynamicsG[:,0], UncertDynamicsG[:,1])
axis[1].set_title("Double Gyre Dynamics")
axis[1].set_xlim([0,2])
axis[1].set_ylim([0,1])
axis[1].set_ylabel("y")
axis[1].legend(["Actual Dynamics", "KNODE Dynamics"])

axis[2].scatter(ActualDynamicsB[:,0], ActualDynamicsB[:,1])
axis[2].scatter(UncertDynamicsB[:,0], UncertDynamicsB[:,1])
axis[2].set_title("Bickley Jet Dynamics")
axis[2].set_xlim([0,20])
axis[2].set_ylim([-3,3])
axis[2].set_ylabel("y")
axis[2].legend(["Actual Dynamics", "KNODE Dynamics"])

plt.show()