import matplotlib.pyplot as plt
from Dataset import GenerateData, GenerateTrainData
import torch
from Dynamics import DuffingOscillator, DoubleGyre, BickleyJet
from KNODE import KNODEDuffing, KNODEDoubleGyre, KNODEBickleyJet, TrainNODENetwork, TestNODENetwork


sizeD, sizeDG, sizeBJ = 40, 200, 400
method = 'dopri5'
batch_time = 10
batch_sizeD, batch_sizeDG, batch_sizeBJ = 25, 18, 25
niters = 2000
test_freq = 10
viz = 'store_true'
adjoint = 'store_true'

DynamicsModel = { "Duffing": 0, "DoubleGyre":1, "BickleyJet":2 }

IC_DG, TrainSizeDG = GenerateData(5, DynamicsModel["DoubleGyre"])
IC_D, TrainSizeD = GenerateData(5, DynamicsModel["Duffing"])
IC_BJ, TrainSizeBJ = GenerateData(5, DynamicsModel["BickleyJet"])

tDG = torch.linspace(0., 20., sizeDG)
tD = torch.linspace(0., 2.75, sizeD)
tBJ = torch.linspace(0., 40., sizeBJ)

DoubleGyreActual = DoubleGyre()
DuffingActual = DuffingOscillator()
BickleyJetActual = BickleyJet()

yTrainDG = GenerateTrainData(TrainSizeDG, sizeDG, IC_DG, tDG, DoubleGyreActual)
yTrainD = GenerateTrainData(TrainSizeD, sizeD, IC_D, tD, DuffingActual)
yTrainBJ = GenerateTrainData(TrainSizeBJ, sizeBJ, IC_BJ, tBJ, BickleyJetActual)

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DuffingKNODE = KNODEDuffing()
DoubleGyreKNODE = KNODEDoubleGyre()
BickleyJetKNODE = KNODEBickleyJet()

DuffingKNODE = TrainNODENetwork(DuffingKNODE, yTrainD, tD, sizeD, batch_time, batch_sizeD)
DoubleGyreKNODE = TrainNODENetwork(DoubleGyreKNODE, yTrainDG, tDG, sizeDG, batch_time, batch_sizeDG)
BickleyJetKNODE = TrainNODENetwork(BickleyJetKNODE, yTrainBJ, tBJ, sizeBJ, batch_time, batch_sizeBJ)

AT, KT = TestNODENetwork(DuffingActual, DuffingKNODE, IC_D[13], tD)

plt.figure()
for i in range(AT.shape[0]):
    plt.scatter(AT[i, 0,0], AT[i, 0,1], s=8, c='b', label='Actual')
    plt.scatter(KT[i, 0,0], AT[i, 0,1], s=8, c='r', label='KNODE')
    plt.legend()
plt.show()