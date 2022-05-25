from QuSim import QuSim as qs

register = qs.QuantumRegister(2)

# Initialise |psi> = 1/sqrt(2) (|0> + |1>)
register.applyGate('X', 1)
register.applyGate('X', 2)

# Entangle qubit 1 & 2
#register.applyGate('CNOT', 1, 2)

# This is a Bell state - maximally entangled
# If qubit 1 is in state |0> then qubit 2
# is also in state |0> Qubit 1 has 50%
# probability of being in |0>

print(register.value)

print(register.measure(qubit=1))

print(register.value)