#### To Do
# Add Toffoli gate
# Create JIRA/Trello workflow
# Add option to simulate or to calculate the state (important in noisy circuits)
# Add Cython implementation

from functools import reduce
import numpy as np

class gates:
    i = np.complex(0, 1)

    singleQubitGates = {
        # Pauli-X / Not Gate
        'X': np.array([
            [0, 1],
            [1, 0]
        ]),
        # Pauli-Y Gate
        'Y': np.array([
            [0, -i],
            [i, 0]
        ]),
        # Pauli-Z Gate
        'Z': np.array([
            [1, 0],
            [0, -1]
        ]),
        # Hadamard Gate
        'H': np.multiply(1. / np.sqrt(2), np.array([
            [1, 1],
            [1, -1]
        ])),
        # Identity Gate
        'Id': np.eye(2),
        # S & S Dagger Gate
        'S': np.array([
            [1, 0],
            [0, i]
        ]),
        'SDagger': np.array([
            [1, 0],
            [0, i]
        ]).conjugate().transpose(),
        # T & T Dagger / Pi over 8 Gate
        'T': np.array([
            [1, 0],
            [0, np.e**(i * np.pi / 4.)]
        ]),
        'TDagger': np.array([
            [1, 0],
            [0, np.e**(i * np.pi / 4.)]
        ]).conjugate().transpose()
    }

    @staticmethod
    def generateGate(gate, numQubits, qubit1, qubit2=1):
        if (gate == 'CNOT'):
            control = qubit1
            target = qubit2

            identity = np.eye(2)
            X = gates.singleQubitGates['X']
            # NaN is our 'C' from the multi qubit gate generation formula
            C = np.array([
                [float('nan'), 0],
                [0, 1]
            ])

            # Set the gate order
            gateOrder = []
            for i in range(1, numQubits + 1):
                if (i == control):
                    gateOrder.append(C)
                elif (i == target):
                    gateOrder.append(X)
                else:
                    gateOrder.append(identity)

            # Generate the gate and then replace the NaNs to Id gates
            newGate = reduce(np.kron, gateOrder)

            n = newGate.shape[0]
            return np.array([[newGate[i, j] if not np.isnan(newGate[i, j]) else 1 if i == j else 0 for j in range(n)] for i in range(n)])

        else:
            # Put these here for handiness
            identity = gates.singleQubitGates['Id']
            mainGate = gates.singleQubitGates[gate]

            # This line of code is not trivial. Loops through all qubits in
            # the register. If
            gateOrder = (mainGate if i == qubit1 else identity
                         for i in range(1, numQubits + 1))
            return reduce(np.kron, gateOrder)


class QuantumRegister:

    def __init__(self, numQubits):
        self.numQubits = numQubits

        # The number of amplitudes needed is 2^N, where N is the
        # number of qubits, So start with a vector of zeros.
        self.amplitudes = np.zeros(2**numQubits)

        # Initialise the state |q0> = |1> in the first qubit
        self.amplitudes.flatten()
        self.amplitudes[0] = 1

        # Reshape the amplitudes so it is human readable and such that
        # factorisation of the state is now trivial.
        # Choosing this implementation over other took a lot of handwritten
        # pseudo-code
        self.amplitudes = self.amplitudes.reshape((2,)*numQubits)

        # Measurement of quantum states is irreversible
        # need a test for whether the qunit has been destroyed
        self.measured = False

        # create another attribute for when measuring a qubit
        # empty initially, as qubits are measured we populate it
        # can remove the need for "measured" attribute.
        # Instead do an assert that the qubit's element is empty
        self.value = np.empty((numQubits))

    def applyGate(self, gate, qubit1, qubit2=-1):
        if self.measured:
            raise ValueError('Cannot Apply Gate to Measured Register')

        elif not self.measured:
            # This means none of the qubits have been measured

            # Generate the gate matrix
            gateMatrix = gates.generateGate(
                gate, self.numQubits, qubit1, qubit2)
            # Calculate the new state vector by multiplying by the gate
            self.amplitudes = np.dot(self.amplitudes.flatten(), gateMatrix).reshape((2,)*self.numQubits)


        elif (self.measured[qubit1-1] or self.measured[qubit2-1]):
            raise ValueError('Cannot Apply Gate to Measured Qubit')


        else:
            # Generate the gate matrix
            gateMatrix = gates.generateGate(
                gate, self.numQubits, qubit1, qubit2)
            # Calculate the new state vector by multiplying by the gate
            self.amplitudes = np.dot(self.amplitudes, gateMatrix)

    def measure(self, qubit=None, cbit=None):
        # If no qubit is given, function measures all qubits
        # else it only measures a single qubit

        self.probabilities = np.zeros(self.numQubits)

        # change measured attribute into an array of bools
        # which has the same dimensions as the register
        # Initialised as False
        self.measured = np.zeros(self.numQubits, dtype=np.bool)

        def bitwise_measure(self, qubit):
            # numpy arrays counter from zero. For human readability
            # we count qubits from 1
            # change measured attribute for this qubit to True
            self.measured[qubit-1] = True

            # np.take factorises the state by 'taking' wrt axis qubit
            amp = np.take(self.amplitudes, 0, axis=(qubit-1))

            # This is the probability the qubit is in state |0>
            p = np.dot(amp.flatten(), amp.conjugate.transpose.flatten())

            print(p)

            # Now, we need to make a weighted random choice of all of the possible
            # output states (done with the range function)

            self.value[qubit-1] = np.random.choice([0, 1], size=1, p=[p, 1-p])


        if qubit==None:
            for i in range(self.numQubits):
                bitwise_measure(self, i)

            return self.value

        if self.measured.all():
            return self.value

        elif self.measure[qubit-1]:
            raise ValueError('Qubit has already been measured')

        else:
            bitwise_measure(self, qubit)
            return self.value[qubit-1]


class ClassicalRegister:

    def __init__(self, numbits):
        self.numbits = numbits

        # Want classical bits which can only be 0 or 1
        # boolean array is a sensible way to impose this
        # Now only have an N dimensional array, not 2^N

        # Unlike the quantum register, we want all states
        # initialised as zero
        self.true = np.zeros(numbits, dtype=np.bool)

        # Classical register only used as an if statement.
        # Do not need any classical logic