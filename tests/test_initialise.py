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
