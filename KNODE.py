<<<<<<< HEAD
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from Dynamics import DoubleGyre, DuffingOscillator, BickleyJet

class KNODEDuffing(DuffingOscillator):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 35),
            nn.Tanh(),
            nn.Linear(35, 35),
            nn.Tanh(),
            nn.Linear(35, 2),
        )
        self.net = self.net.float()
        self.net.apply(self._apply_wt_init)

    def forward(self, t, y):
        NNOut = self.net(y)
        KOut = self.DuffingOscillatorDynamics(y.T, t, self.Alpha, self.Beta, self.Delta)
        b = torch.vstack((KOut[0], KOut[1])).T
        return b + NNOut
    
    def _apply_wt_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, val=0)

class KNODEDoubleGyre(DoubleGyre):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 35),
            nn.Tanh(),
            nn.Linear(35, 35),
            nn.Tanh(),
            nn.Linear(35, 2),
        )
        self.net = self.net.float()
        self.net.apply(self._apply_wt_init)

    def forward(self, t, y):
        NNOut = self.net(y)
        KOut = self.DoubleGyreDynamics(y, t, self.eps, self.alpha, self.A, self.omega)
        b = torch.vstack((KOut[0], KOut[1])).T
        return b + NNOut
    
    def _apply_wt_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, val=0)

class KNODEBickleyJet(BickleyJet):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 35),
            nn.Tanh(),
            nn.Linear(35, 35),
            nn.Tanh(),
            nn.Linear(35, 2),
        )
        self.net = self.net.float()
        self.net.apply(self._apply_wt_init)

    def forward(self, t, y):
        NNOut = self.net(y)
        KOut = self.BickleyJetDynamics(y, t, self.U0, self.L0, self.eps1, self.eps2, self.eps3, self.R0, self.Dphix, self.Dphiy)
        b = torch.vstack((KOut[0], KOut[1])).T
        return b + NNOut

    def _apply_wt_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, val=0)

def mini_batch(y, t, size, time, batch_size):
        
    s = torch.from_numpy(np.random.choice(np.arange(size-time, dtype=np.int64), batch_size, replace=False))
    k = torch.randint(0,y.shape[0],(1,))
    batch_y0 = torch.tensor(y[k,s,:])
    batch_t = torch.tensor(t[:time])
    batch_y = torch.stack([torch.Tensor(y[k,s + i,:]) for i in range(time)], dim=0)

    return batch_y0, batch_t, batch_y

def TrainNODENetwork(Model, y, t, size, time, batch_size, niters=300):

    optimizer = optim.Adam(Model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    for itr in range(1,niters +1):

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = mini_batch(y, t, size, time, batch_size)
        pred_y = odeint(Model, batch_y0, batch_t)

        loss = torch.mean(criterion(pred_y, batch_y))
        loss.backward()
        optimizer.step()

        if itr % 10 == 0:
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
    
    return Model

def TestNODENetwork(ModelAct, ModelKnode, IC, t):

    y0 = torch.reshape(torch.tensor([IC[0], IC[1]]).float(), (1,2))
    ActTrajectory = odeint(ModelAct, y0, t, method='dopri5').detach().numpy()
    KNODETrajectory = odeint(ModelKnode, y0, t).detach().numpy()
=======
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from Dynamics import DoubleGyre, DuffingOscillator, BickleyJet

class KNODEDuffing(DuffingOscillator):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 35),
            nn.Tanh(),
            nn.Linear(35, 35),
            nn.Tanh(),
            nn.Linear(35, 2),
        )
        self.net = self.net.float()
        self.net.apply(self._apply_wt_init)

    def forward(self, t, y):
        NNOut = self.net(y)
        KOut = self.DuffingOscillatorDynamics(y.T, t, self.Alpha, self.Beta, self.Delta)
        b = torch.vstack((KOut[0], KOut[1])).T
        return b + NNOut
    
    def _apply_wt_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, val=0)

class KNODEDoubleGyre(DoubleGyre):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 35),
            nn.Tanh(),
            nn.Linear(35, 35),
            nn.Tanh(),
            nn.Linear(35, 2),
        )
        self.net = self.net.float()
        self.net.apply(self._apply_wt_init)

    def forward(self, t, y):
        NNOut = self.net(y)
        KOut = self.DoubleGyreDynamics(y, t, self.eps, self.alpha, self.A, self.omega)
        b = torch.vstack((KOut[0], KOut[1])).T
        return b + NNOut
    
    def _apply_wt_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, val=0)

class KNODEBickleyJet(BickleyJet):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 35),
            nn.Tanh(),
            nn.Linear(35, 35),
            nn.Tanh(),
            nn.Linear(35, 2),
        )
        self.net = self.net.float()
        self.net.apply(self._apply_wt_init)

    def forward(self, t, y):
        NNOut = self.net(y)
        KOut = self.BickleyJetDynamics(y, t, self.U0, self.L0, self.eps1, self.eps2, self.eps3, self.R0, self.Dphix, self.Dphiy)
        b = torch.vstack((KOut[0], KOut[1])).T
        return b + NNOut

    def _apply_wt_init(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.1)
            nn.init.constant_(layer.bias, val=0)

def mini_batch(y, t, size, time, batch_size):
        
    s = torch.from_numpy(np.random.choice(np.arange(size-time, dtype=np.int64), batch_size, replace=False))
    k = torch.randint(0,y.shape[0],(1,))
    batch_y0 = torch.tensor(y[k,s,:])
    batch_t = torch.tensor(t[:time])
    batch_y = torch.stack([torch.Tensor(y[k,s + i,:]) for i in range(time)], dim=0)

    return batch_y0, batch_t, batch_y

def TrainNODENetwork(Model, y, t, size, time, batch_size, niters=300):

    optimizer = optim.Adam(Model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    for itr in range(1,niters +1):

        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = mini_batch(y, t, size, time, batch_size)
        pred_y = odeint(Model, batch_y0, batch_t)

        loss = torch.mean(criterion(pred_y, batch_y))
        loss.backward()
        optimizer.step()

        if itr % 10 == 0:
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
    
    return Model

def TestNODENetwork(ModelAct, ModelKnode, IC, t):

    y0 = torch.reshape(torch.tensor([IC[0], IC[1]]).float(), (1,2))
    ActTrajectory = odeint(ModelAct, y0, t, method='dopri5').detach().numpy()
    KNODETrajectory = odeint(ModelKnode, y0, t).detach().numpy()
>>>>>>> 98cab118871594355766d015a467f566878df94d
    return ActTrajectory, KNODETrajectory