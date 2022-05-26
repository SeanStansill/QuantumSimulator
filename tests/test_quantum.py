# This test checks that we can import our module

# Note we do not need to define the abolute/relative path as this is
# done in pytest.ini

import QuSim
import numpy as np

def test_qubit_register():
    q1 = QuSim.QuantumRegister(2)

    # We want to test that this is a simple numpy array
    # Do not need to check here the contents of the array
    # Simply check that it exists and is the correct shape
    assert np.shape(q1._get_amplitudes()) == np.shape(np.zeros((2, 2)))



# In this test we wish to create a quantum register
# then create a superposition of states and measure the outcome

# We must test the result many times since then outcome is probabilistic
# We could equally test the amplitude is correct. However, this tests
# both the apply_gate and measure methods simultaneously

def test_entanglement():

    # apply this test 100 times to ensure all outcomes appear at least once
    # and that there is no small probability of error (this is a noiseless system)
    for i in range(100):

        q1 = QuSim.QuantumRegister(2)

        # Create a Bell state
        q1.applyGate('H', 1)
        q1.applyGate('CNOT', 1, 2)

        # Measure the system
        q1.measure()

        # We expect both particles to be |0> or both |1>
        assert (q1.value.all() == 1.0 or q1.value.all() == 0.0)




def test_three_qubit_entanglement():

    # apply this test 100 times to ensure all outcomes appear at least once
    # and that there is no small probability of error (this is a noiseless system)
    for i in range(100):

        q1 = QuSim.QuantumRegister(3)

        # Create a Bell state
        q1.applyGate('H', 1)
        q1.applyGate('CNOT', 1, 2)
        q1.applyGate('CNOT', 2, 3)

        # Measure the system
        q1.measure()

        # We expect both particles to be |0> or both |1>
        assert (q1.value.all() == 1.0 or q1.value.all() == 0.0)


def test_single_qubit_measurement():
    # applies the test 100 times to check that a partial measurement returns the correct probability amplitudes
    for i in range(100):
        qr = QuSim.QuantumRegister(2)

        qr.applyGate('H', 1)
        qr.applyGate('CNOT', 1, 2)

        q1 = qr.measure(1)
        print(qr.amplitudes[qr._get_slice(0,1)])
        if q1:
            assert np.array_equal(qr.amplitudes[qr._get_slice(0,0)], [0.0+0.0j, 0.0+0.0j]) and np.array_equal(qr.amplitudes[qr._get_slice(0,1)], [0.0+0.0j, 1.0+0.0j])

        if not q1:
            assert np.array_equal(qr.amplitudes[qr._get_slice(1,0)], [1.0+0.0j, 0.0+0.0j]) and np.array_equal(qr.amplitudes[qr._get_slice(0,1)], [0.0+0.0j, 0.0+0.0j])

def test_single_qubit_measurement2():
    # Apply Deutsch's algorithm to a randomly generated 2-qubit state
    for i in range(100):
        qr = QuSim.QuantumRegister(2)

        # Initialize in either |01> or |10>
        # Pick a random number which is either 1 or 2
        # denoting the first or second qubit which is chosen as
        # state |1> (this is equivalent to the choice of "which
        # hand is the coin in")
        qubit = np.random.choice([1, 2], size=1, p=[0.5, 0.5])

        # Apply the X gate to the random qubit chosen
        qr.applyGate('X', qubit)

        # Apply Deutsch's algorithm
        qr.applyGate('H', 1)
        qr.applyGate('H', 2)

        qr.applyGate('CNOT', 1, 2)

        qr.applyGate('H', 1)
        qr.applyGate('H', 2)

        # Now the first qubit should be in state |1>
        # If state of first qubit given by |psi_1> = a|0> + b|1>,
        # b contains information about all other bits (so is an array)
        # this is given as
        q1_amp_one = qr.amplitudes[qr._get_slice(0, 1)]

        # Calculate probability state is in |1> by calculating
        # b \cdot b^{\dagger}

        np.testing.assert_approx_equal(np.sum(q1_amp_one * q1_amp_one.conj().T), 1.0, significant=3)