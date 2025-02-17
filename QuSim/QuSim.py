#### To Do
# Add Toffoli gate
# Create JIRA/Trello workflow
# Add option to simulate the circuit or to calculate the density matrix (important in noisy circuits)
# Add Cython implementation
# Add unit testing

from functools import reduce
import numpy as np

class gates:
    # We are safe to define the complex number as a variable
    # because of the scope of the class
    i = complex(0, 1)

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
        # np.e is much faster than np.exp when we know the argument is a float not an array
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
        self.amplitudes = np.zeros(2**numQubits, dtype=complex)

        # Initialise the state where all qubits in |0>
        self.amplitudes.flatten()
        self.amplitudes[0] = 1

        # Reshape the amplitudes so it is human readable and such that
        # factorisation of the state is now trivial.
        # Choosing this implementation over other took a lot of handwritten
        # pseudo-code
        self.amplitudes = self.amplitudes.reshape((2,)*numQubits)

        # Measurement of quantum states is irreversible
        # need a test for whether the qunit has been destroyed
        self.measured = np.zeros(self.numQubits, dtype=bool)

        # create another attribute for when measuring a qubit
        # empty initially, as qubits are measured we populate it
        # can remove the need for "measured" attribute.
        # Instead do an assert that the qubit's element is empty
        self.value = np.empty((numQubits))

    # Most useful as an external function
    def get_amplitudes(self, qubit=None, state=0):
        if qubit == None:
            return self.amplitudes

        else:
            return self.amplitudes[self._get_slice(qubit, state)]


    def _get_slice(self, qubit, state=0):
        idx = [slice(None)] * self.numQubits

        idx[qubit] = state

        return tuple(idx)

    def _get_all_other_slices(self, qubit):
        idx = [slice(None)] * self.numQubits

        for i in range(self.numQubits):
            if i == qubit:
                continue

            else:
                idx[i] = not int(self.value[i])

        return tuple(idx)

    def _get_qubit_amplitude(self, qubit):
        qubit_state_zero = self._get_slice(qubit, 0)
        qubit_state_one = self._get_slice(qubit, 1)

        return [self.get_amplitudes(qubit, 0), self.get_amplitudes(qubit, 1)]


    def _collapse_state(self, qubit):

        # Change the qubit amplitudes based on the value of the measured qubit
        if self.value[qubit]:

            # For qubit |psi_i> = a|0> + b|1>

            # Qubit has been measured to be in state |1>, set a = 0
            self.amplitudes[self._get_slice(qubit, 0)] = np.zeros(np.shape(self.amplitudes[self._get_slice(qubit, 0)]), dtype=complex)

            # Now normalise b s.t |b|=1
            # First get and store b
            amp = self.amplitudes[self._get_slice(qubit, 1)]

            # b is not a simple scalar as it has all other qubit states factored
            # normalisation factor is amp \cdot amp^{\dagger}
            normalisation_factor = np.sum(amp @ amp.conj().T)**0.5

            # Normalise the amplitudes
            self.amplitudes[self._get_slice(qubit, 1)] *= 1/normalisation_factor

        elif not self.value[qubit]:

            # For qubit |psi_i> = a|0> + b|1>

            # Qubit has been measured to be in state |0>, set b = 0
            self.amplitudes[self._get_slice(qubit, 1)] = np.zeros(np.shape(self.amplitudes[self._get_slice(qubit, 1)]), dtype=complex)

            # Now normalise b s.t |a|=1
            # First get and store a
            amp = self.amplitudes[self._get_slice(qubit, 0)]

            # a is not a simple scalar as it has all other qubit states factored
            # normalisation factor is amp \cdot amp^{\dagger}
            normalisation_factor = np.real(np.sum(amp @ amp.conj().T)**0.5)

            # Normalise the amplitudes
            self.amplitudes[self._get_slice(qubit, 0)] *= 1/normalisation_factor



    def applyGate(self, gate, qubit1, qubit2=None, control=True):

        # if only one qubit has been given, we only need to check one qubit
        if qubit2==None:

            # check qubit hasn't been measured
            if self.measured[qubit1-1]:
                raise ValueError('Cannot Apply Gate to Measured Qubits')

            else:
                # Generate the gate matrix
                gateMatrix = gates.generateGate(
                    gate, self.numQubits, qubit1)
                # Calculate the new state vector by multiplying by the gate
                self.amplitudes = np.dot(self.amplitudes.flatten(), gateMatrix).reshape((2,) * self.numQubits)

        else:
            if (self.measured[qubit1-1] or self.measured[qubit2-1]):
                raise ValueError('Cannot Apply Gate to Measured Qubits')

            elif bool(control)==True:
                # This means none of the qubits have been measured

                # Generate the gate matrix
                gateMatrix = gates.generateGate(
                    gate, self.numQubits, qubit1, qubit2)
                # Calculate the new state vector by multiplying by the gate
                self.amplitudes = np.dot(self.amplitudes.flatten(), gateMatrix).reshape((2,)*self.numQubits)


    def measure(self, qubit=None, cbit=None):

        # First if a qubit is given, move from human readable
        # to machine indexing (ie count from 0 instead of 1)
        # we only need to do this once
        if qubit != None:
            qubit -= 1

        def bitwise_measure(self, qubit):
            # numpy arrays counter from zero. For human readability
            # we count qubits from 1
            # change measured attribute for this qubit to True
            self.measured[qubit] = True

            amp = self.get_amplitudes(qubit, 0)

            # This is the probability the qubit is in state |0>
            probability = np.sum(amp@amp.conj().T)

            # Now, we need to make a weighted random choice of all of the possible
            # output states (done with the range function)
            self.value[qubit] = np.random.choice([0, 1], size=1, p=[probability.real, 1.0-probability.real])

            if self.measured.all():
                pass
            else:
                self._collapse_state(qubit)

        if qubit==None:
            for i in range(self.numQubits):
                bitwise_measure(self, i)

            return self.value

        elif self.measured[qubit]:
            # Choose to re-return the measured value instead of raising an exception
            return self.value[qubit]

        else:
            bitwise_measure(self, qubit)
            return self.value[qubit]


class ClassicalRegister:

    def __init__(self, numbits):
        self.numbits = numbits

        # Want classical bits which can only be 0 or 1
        # boolean array is a sensible way to impose this
        # Now only have an N dimensional array, not 2^N

        # Unlike the quantum register, we want all states
        # initialised as zero
        self.true = np.zeros(numbits, dtype=bool)

        # Classical register only used as an if statement.
        # Do not need any classical logic