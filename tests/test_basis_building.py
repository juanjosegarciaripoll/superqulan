import unittest
from superqulan.bosons import construct_basis, Basis, State


class TestBosonicBasis(unittest.TestCase):
    def assertEqualBasis(self, basis: Basis, state_list: list[State]):
        """Verify that the basis contains the given list of configurations."""
        self.assertTrue(len(basis) == len(state_list))
        self.assertTrue(all(s in basis for s in state_list))

    def assertIdenticalBasis(self, basis: Basis, sorted_state_list: list[State]):
        """Verify that the basis contains the given list of configurations."""
        self.assertTrue(len(basis) == len(sorted_state_list))
        other_basis = {
            state: position for position, state in enumerate(sorted_state_list)
        }
        self.assertTrue(basis == other_basis)

    def assertEqualExcitations(self, basis: Basis, excitations: int):
        """Verify that all states have the same number of `excitations`"""
        self.assertTrue(all(len(a) == excitations for a in basis))

    def assertBasisContainsAllIndices(self, basis: Basis):
        """Verify that the basis contains indices to all states in the basis,
        without duplicates."""
        L = len(basis)
        indices = sorted(b for _, b in basis.items())
        self.assertTrue(all(i == j for i, j in zip(indices, range(L))))

    def assertModesAreSorted(self, basis: Basis):
        """Verify that the basis contains indices to all states in the basis,
        without duplicates."""
        self.assertTrue(all(s == tuple(sorted(s)) for s in basis))

    def run_for_all_combinations(self, function, excitation_range=[1], max_modes=4):
        for excitations in excitation_range:
            for modes in range(max_modes + 1):
                for qubits in range(modes):
                    basis = construct_basis(
                        qubits=qubits, bosons=modes - qubits, excitations=excitations
                    )
                    function(modes, excitations, basis)

    def test_qubits_and_bosons_same_for_1_excitation(self):
        for modes in range(4):
            last = None
            for qubits in range(modes):
                basis = construct_basis(
                    qubits=qubits, bosons=modes - qubits, excitations=1
                )
                if last is not None:
                    self.assertEqualBasis(basis, last)

    def test_1_qubit_0_bosons_0_excitations(self):
        one_qubit_0 = [()]
        self.assertEqualBasis(
            construct_basis(qubits=1, bosons=0, excitations=0), one_qubit_0
        )

    def test_1_qubit_0_bosons_1_excitations(self):
        one_qubit_1 = [(0,)]
        self.assertEqualBasis(
            construct_basis(qubits=1, bosons=0, excitations=1), one_qubit_1
        )

    def test_0_qubits_1_bosons_0_excitations(self):
        one_boson_0 = [()]
        self.assertEqualBasis(
            construct_basis(qubits=0, bosons=1, excitations=0), one_boson_0
        )

    def test_0_qubits_1_bosons_1_excitations(self):
        one_boson_1 = [(0,)]
        self.assertEqualBasis(
            construct_basis(qubits=0, bosons=1, excitations=1), one_boson_1
        )

    def test_1_qubit_1_boson_2_excitations(self):
        self.assertEqualBasis(
            construct_basis(qubits=1, bosons=1, excitations=2), [(0, 1), (1, 1)]
        )

    def test_1_qubit_2_boson_2_excitations(self):
        self.assertEqualBasis(
            construct_basis(qubits=1, bosons=2, excitations=2),
            [(0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
        )

    def test_2_qubit_2_boson_2_excitations(self):
        self.assertIdenticalBasis(
            construct_basis(qubits=2, bosons=2, excitations=2),
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
        )

    def test_zero_excitation_basis_is_trivial(self):
        """For zero excitations, we only create the vacuum state, represented by
        an empty tuple."""
        self.run_for_all_combinations(
            lambda modes, excitations, basis: self.assertIdenticalBasis(basis, [()]),
            excitation_range=[0],
            max_modes=100,
        )

    def test_one_excitation_basis_is_sorted(self):
        """The basis for 1 excitation is a map between the occupied mode (mode,) and
        the index to that mode, which is the number 'mode' itself"""
        self.run_for_all_combinations(
            lambda modes, excitations, basis: self.assertIdenticalBasis(
                basis, list((i,) for i in range(modes))
            ),
            excitation_range=[1],
            max_modes=100,
        )

    def test_number_of_excitations_is_right(self):
        def test(modes, excitations, basis):
            self.assertEqualExcitations(basis, excitations)
            self.assertBasisContainsAllIndices(basis)

        self.run_for_all_combinations(test, excitation_range=range(5), max_modes=4)

    def test_basis_states_contains_sorted_modes(self):
        self.run_for_all_combinations(
            lambda modes, excitations, basis: self.assertModesAreSorted(basis),
            excitation_range=range(5),
            max_modes=4,
        )
