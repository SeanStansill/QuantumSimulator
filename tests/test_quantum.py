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
    assert np.shape(q1.get_amplitudes()) == np.shape(np.zeros((2, 2)))



# In this test we wish to create a quantum register
# then create a superposition of states and measure the outcome

# We must test the result many times since then outcome is probabilistic
# We could equally test the amplitude is correct. However, this tests
# both the apply_gate and measure methods simultaneously

def test_entanglement():

    # apply this test 100 times
    for i in range(100):

        q1 = QuSim.QuantumRegister(2)

        q1.applyGate('H', 1)

        q1.applyGate('CNOT', 1, 2)
        q1.measure()

        assert (q1.value == np.array([1.0, 1.0]) or q1.value == np.array([0.0, 0.0]))




def two_qubit_collapse():
    pass