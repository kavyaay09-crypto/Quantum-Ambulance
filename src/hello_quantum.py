from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# 2 qubits: put qubit 0 in superposition, then entangle qubit 1 with it
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

sim = AerSimulator()
result = sim.run(qc, shots=1000).result()
print(result.get_counts())
