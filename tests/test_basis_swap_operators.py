import unittest
import scipy.sparse as sp
from superqulan.bosons import construct_basis, move_excitation_operator, number_operator
from .common import *


class TestExcitationMotion(unittest.TestCase):
    def run_for_all_combinations(
        self, function, excitation_range=[1], max_modes=4, max_qubits=None
    ):
        if max_qubits is None:
            max_qubits = max_modes
        for excitations in excitation_range:
            for modes in range(max_modes + 1):
                for qubits in range(max_qubits):
                    basis = construct_basis(
                        qubits=qubits, bosons=modes - qubits, excitations=excitations
                    )
                    for i in range(modes):
                        for j in range(modes):
                            function(modes, excitations, i, j, basis)

    def test_motion_operator_is_trivial_for_0_excitations(self):
        """When there are zero excitations, no motion is possible."""

        def test(modes, excitations, i, j, basis):
            L = len(basis)
            O = move_excitation_operator(origin_mode=i, destination_mode=j, basis=basis)
            self.assertTrue(is_zero_sparse_matrix(O))
            self.assertTrue(O.shape == (L, L))

        self.run_for_all_combinations(test, excitation_range=[0], max_modes=10)

    def test_exact_motion_operator_for_1_excitations(self):
        """When working with 1 excitation, the basis is an ordered set of states,
        each one corresponding to one occupied mode. The a_j^+ a_i operator is
        then a matrix with O[j,i] = 1.0 and zeros elsewhere."""

        def test(modes, excitations, i, j, basis):
            L = len(basis)
            O = move_excitation_operator(origin_mode=i, destination_mode=j, basis=basis)
            Oexact = sp_from_coordinates([(1.0, j, i)], shape=(L, L))
            self.assertTrue(equal_sparse_matrices(O, Oexact))

        self.run_for_all_combinations(test, excitation_range=[1], max_modes=10)

    def test_cannot_move_twice_into_qubit(self):
        basis = construct_basis(qubits=3, bosons=3, excitations=3)
        for qubit_mode in range(3):
            for boson_mode in range(3, 6):
                Oji = move_excitation_operator(
                    origin_mode=boson_mode, destination_mode=qubit_mode, basis=basis
                )
                self.assertTrue(is_zero_sparse_matrix(Oji @ Oji))

    def test_cannot_move_twice_out_of_qubit(self):
        basis = construct_basis(qubits=3, bosons=3, excitations=3)
        for qubit_mode in range(3):
            for boson_mode in range(3, 6):
                Oji = move_excitation_operator(
                    origin_mode=qubit_mode, destination_mode=boson_mode, basis=basis
                )
                self.assertTrue(is_zero_sparse_matrix(Oji @ Oji))

    def test_back_and_forth_operations(self):
        """The combination of i->j and j->i is a product of number operators."""

        def test(modes, excitations, i, j, basis):
            Oji = move_excitation_operator(
                origin_mode=i, destination_mode=j, basis=basis
            )
            Oij = move_excitation_operator(
                origin_mode=j, destination_mode=i, basis=basis
            )
            Ni = number_operator(basis, i)
            Nj = number_operator(basis, j)
            id = sp.eye(Nj.shape[0])
            if i == j:
                self.assertTrue(close_sparse_matrices(Oij @ Oji, Ni @ Ni))
            else:
                self.assertTrue(close_sparse_matrices(Oij @ Oji, (Nj + id) @ Ni))

        self.run_for_all_combinations(
            test, excitation_range=[2], max_modes=10, max_qubits=0
        )
