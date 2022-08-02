import torch
import pennylane as qml

torch.random.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device.'.format(device))

dev_u = qml.device('default.qubit', wires=1)

def SX(x):
    qml.RX(x, wires=0)

def WU(theta):
    qml.Rot(theta[0], theta[1], theta[2], wires=0)

@qml.qnode(dev_u, interface='torch')
def unimodal_processing_circuit(inputs, weights):
    for t in range(len(inputs)):
        WU(weights[t])
        SX(inputs[t])

    WU(weights[-1])
    return qml.expval(qml.PauliZ(wires=0))

weight_shapes = {'weights': (21, 3)}
upc_a = qml.qnn.TorchLayer(unimodal_processing_circuit, weight_shapes).to(device)
upc_b = qml.qnn.TorchLayer(unimodal_processing_circuit, weight_shapes).to(device)

x = torch.rand((32, 20), dtype=torch.float32).to(device)
print('x device: {}'.format(x.device))

res = upc_a(x)
print('res dtype: {}'.format(res.dtype))
print('res shape: {}'.format(res.shape))
print('res device: {}'.format(res.device))
