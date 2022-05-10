#### To Do
# Add Toffoli gate
# Create JIRA/Trello workflow

from functools import reduce
import numpy as np

class gates:
    i = np.complex(0, 1)

    singleQubitGates = {
        # Pauli-X / Not Gate
        'X': np.matrix([
            [0, 1],
            [1, 0]
        ]),
        # Pauli-Y Gate
        'Y': np.matrix([
            [0, -i],
            [i, 0]
        ]),
        # Pauli-Z Gate
        'Z': np.matrix([
            [1, 0],
            [0, -1]
        ]),
        # Hadamard Gate
        'H': np.multiply(1. / np.sqrt(2), np.matrix([
            [1, 1],
            [1, -1]
        ])),
        # Identity Gate
        'Id': np.eye(2),
        # S & S Dagger Gate
        'S': np.matrix([
            [1, 0],
            [0, i]
        ]),
        'SDagger': np.matrix([
            [1, 0],
            [0, i]
        ]).conjugate().transpose(),
        # T & T Dagger / Pi over 8 Gate
        'T': np.matrix([
            [1, 0],
            [0, np.e**(i * np.pi / 4.)]
        ]),
        'TDagger': np.matrix([
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
            C = np.mat([
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
            return np.mat([[newGate[i, j] if not np.isnan(newGate[i, j]) else 1 if i == j else 0 for j in range(n)] for i in range(n)])

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

        # Initialise the state |psi> = |1> in the first qubit
        self.amplitudes[0] = 1

        # Measurement of quantum states is irreversible
        # need a test for whether the qunit has been destroyed
        self.measured = False


    def applyGate(self, gate, qubit1, qubit2=-1):
        if self.measured:
            raise ValueError('Cannot Apply Gate to Measured Register')

        elif not self.measured:
            # This means none of the qubits have been measured

            # Generate the gate matrix
            gateMatrix = gates.generateGate(
                gate, self.numQubits, qubit1, qubit2)
            # Calculate the new state vector by multiplying by the gate
            self.amplitudes = np.dot(self.amplitudes, gateMatrix)


        elif (self.measured[qubit1-1] or self.measured[qubit2-1]):
            raise ValueError('Cannot Apply Gate to Measured Qubit')


        else:
            # Generate the gate matrix
            gateMatrix = gates.generateGate(
                gate, self.numQubits, qubit1, qubit2)
            # Calculate the new state vector by multiplying by the gate
            self.amplitudes = np.dot(self.amplitudes, gateMatrix)

    def measure(self, qubit=None):
        # If no qubit is given, function measures all qubits
        # else it only measures a single qubit

        self.probabilities = np.zeros(self.numQubits)

        # change measured attribute into an array of bools
        # which has the same dimensions as the register
        # Initialised as False
        self.measured = np.zeros(self.numQubits, dtype=np.bool)

        def bitwise_measure(qubit):
            # numpy arrays counter from zero. For human readability
            # we count qubits from 1
            self.measured[qubit-1] = True
            # Get this list of probabilities, by squaring the absolute
            # value of the amplitudes
            self.probabilities = []
            for amp in np.nditer(self.amplitudes):
                probability = np.absolute(amp)**2
                self.probabilities.append(probability)

            # Now, we need to make a weighted random choice of all of the possible
            # output states (done with the range function)

            results = list(range(len(self.probabilities)))
            self.value = np.binary_repr(
                np.random.choice(results, p=self.probabilities),
                self.numQubits
            )
            return self.value


        if qubit==None:
            for i in range(self.numQubits):
                bitwise_measure(i)

        if self.measured.all():
            return self.measure
        else:
            # Get this list of probabilities, by squaring the absolute
            # value of the amplitudes
            self.probabilities = []
            for amp in np.nditer(self.amplitudes):
                probability = np.absolute(amp)**2
                self.probabilities.append(probability)

            # Now, we need to make a weighted random choice of all of the possible
            # output states (done with the range function)

            results = list(range(len(self.probabilities)))
            self.value = np.binary_repr(
                np.random.choice(results, p=self.probabilities),
                self.numQubits
            )
            return self.value


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