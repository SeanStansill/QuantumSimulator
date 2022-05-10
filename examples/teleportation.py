import QuSim as qs

register = qs.QuantumRegister(3)

register.applyGate('X', 1)

register.applyGate('H', 2)

register.applyGate('CNOT', 2, 3)

register.applyGate('CNOT', 1, 2)

register.applyGate('H', 1)

print(QuSim.measure(register))