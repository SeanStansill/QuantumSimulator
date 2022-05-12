import QuSim as qs

register = qs.QuantumRegister(3)

# Initialise |psi> = 1/sqrt(2) (|0> + |1>)
register.applyGate('X', 1)

# teleportation algorithm
register.applyGate('H', 2)

register.applyGate('CNOT', 2, 3)

register.applyGate('CNOT', 1, 2)

register.applyGate('H', 1)

register.measure(2)

print(register.measure())