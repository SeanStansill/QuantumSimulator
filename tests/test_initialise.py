# This test checks that we can import our module

# Note we do not need to define the abolute/relative path as this is
# done in pytest.ini

import QuSim

def test_qubit_register():
    q1 = QuSim.QuantumRegister(2)

