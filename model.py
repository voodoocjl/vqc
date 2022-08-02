import torch
from torch import nn
import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers

"""Unimodal processing quantum circuits"""
dev_u = qml.device('default.qubit', wires=1)

def SX(x):
    qml.RX(x, wires=0)

def WU(theta):
    qml.Rot(theta[0], theta[1], theta[2], wires=0)

@qml.qnode(dev_u, interface='torch')
def unimodal_processing_circuit(inputs, weights):
    """
    Args:
      inputs: unimodal feature of each time step in an example, size (N, T)
      weights: parameters of trainable circuit block, size (T+1, 3)

    Returns:
      expectation value of PauliZ observable
    """
    for t in range(len(inputs)):
        WU(weights[t])
        SX(inputs[t])

    WU(weights[-1])
    return qml.expval(qml.PauliZ(wires=0))

"""Multimodal fusion quantum circuit"""
def multimodal_fusion_vqc(n_ansatz_layers, n_fusion_layers):
    dev_m = qml.device('default.qubit', wires=3)

    def S(x):
        for m in range(3):
            qml.RX(x[m], wires=m)

    def WM(theta):
        StronglyEntanglingLayers(theta, wires=[0, 1, 2])

    def multimodal_fusion_circuit(inputs, weights):
        """
        Args:
          inputs: unimodal feature of each time step in an example, size (N, 3)
          weights: parameters of trainable circuit block, size (n_fusion_layers+1, n_ansatz_layers, 3, 3)

        Returns:
          expectation value of PauliZ observable in wire 0
        """
        for l in range(n_fusion_layers):
            WM(weights[l])
            S(inputs)
        WM(weights[-1])
        return qml.expval(qml.PauliZ(wires=0))

    qnode = qml.QNode(multimodal_fusion_circuit, dev_m, interface='torch')
    weight_shapes = {'weights': (n_fusion_layers+1, n_ansatz_layers, 3, 3)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    return qlayer

"""Hybrid quantum-classical fusion network"""
class HQCFN(nn.Module):
    def __init__(self, n_time_step, input_dims, n_ansatz_layers, n_fusion_layers):
        super(HQCFN, self).__init__()
        self.n_time_step = n_time_step
        self.input_dims = input_dims
        self.n_ansatz_layers = n_ansatz_layers
        self.n_fusion_layers = n_fusion_layers

        self.linear_a = nn.Linear(input_dims[0], 1)
        self.linear_v = nn.Linear(input_dims[1], 1)
        self.linear_t = nn.Linear(input_dims[2], 1)

        self.unimodal_a = qml.qnn.TorchLayer(
            unimodal_processing_circuit, {'weights': (n_time_step+1, 3)})
        self.unimodal_v = qml.qnn.TorchLayer(
            unimodal_processing_circuit, {'weights': (n_time_step+1, 3)})
        self.unimodal_t = qml.qnn.TorchLayer(
            unimodal_processing_circuit, {'weights': (n_time_step+1, 3)})

        self.multimodal = multimodal_fusion_vqc(n_ansatz_layers, n_fusion_layers)

    def forward(self, x):
        x_a = self.linear_a(x[0]).squeeze()
        x_a = nn.functional.relu(x_a, inplace=True)
        x_v = self.linear_v(x[1]).squeeze()
        x_v = nn.functional.relu(x_v, inplace=True)
        x_t = self.linear_t(x[2]).squeeze()
        x_t = nn.functional.relu(x_t, inplace=True)

        expval_a = self.unimodal_a(x_a)
        expval_v = self.unimodal_v(x_v)
        expval_t = self.unimodal_t(x_t)
        expvals = torch.stack([expval_a, expval_v, expval_t], dim=1)

        pred = self.multimodal(expvals)
        return pred
