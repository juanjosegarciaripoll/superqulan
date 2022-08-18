from re import I
from typing import Optional, Union, Callable
from dataclasses import dataclass, field
import wave
from numpy.typing import NDArray, ArrayLike
from math import pi as π
from numbers import Number
import numpy as np
import scipy.sparse as sp
from .waveguide import Waveguide
from .bosons import (
    Basis,
    State,
    construct_basis,
    mode_occupations,
    move_excitation_operator,
    diagonals_with_energies,
    concatenate_basis,
)


@dataclass
class Setup:
    """Setup is an abstract class for all setups that combine qubits, cavities
    and waveguides. It consists of the following input fields

    Nexcitations (int):         Maximum number of excitations considered in the basis
    number_conserving (bool):   Whether the basis includes vacuum and other lower states

    Nqubits (int):              Number of qubits in the setup
    Ncavities (int):            Number of cavities in the setup
    Nfilters (int):             Number of Purcell filters in the setup
    waveguide:                  If supplied, the Waveguide object describing those modes.
                                Otherwise the derived class must define a default_waveguide()
                                method that computes this object.

    The class has the following computed fields:

    qubit_indices:              List of mode numbers associated to the qubits
    cavity_indices:             List of mode numbers associated to the cavities
    filter_indices:             List of mode numbers associated to the Purcell filters
    waveguide_indices:          List of mode numbers associated to the waveguide modes
    size:                       Total number of states in the basis
    basis (Basis):              Bosonic basis as defined in superqulan.bosons
    """

    number_conserving: bool = False
    Nexcitations: int = 1

    Nqubits: int = 2
    Ncavities: int = 2
    Nfilters: int = 0
    waveguide: Optional[Waveguide] = None

    qubit_indices: list[int] = field(init=False)
    cavity_indices: list[int] = field(init=False)
    filter_indices: list[int] = field(init=False)
    waveguide_indices: list[int] = field(init=False)

    size: int = field(init=False)
    basis: Basis = field(init=False)

    def __post_init__(self):

        if self.waveguide is None:
            self.waveguide = self.default_waveguide()

        self.qubit_indices = range(0, self.Nqubits)
        self.cavity_indices = range(self.Nqubits, self.Nqubits + self.Ncavities)
        self.filter_indices = range(
            self.Nqubits + self.Ncavities, self.Nqubits + self.Ncavities + self.Nfilters
        )
        waveguide_start = self.Nqubits + self.Ncavities + self.Nfilters
        self.waveguide_indices = range(
            waveguide_start, waveguide_start + self.waveguide.modes
        )

        if self.number_conserving:
            self.basis = concatenate_basis(
                qubits=self.Nqubits,
                bosons=self.Ncavities + self.Nfilters + self.waveguide.modes,
                excitations=self.Nexcitations,
            )
        else:
            self.basis = construct_basis(
                qubits=self.Nqubits,
                bosons=self.Ncavities + self.Nfilters + self.waveguide.modes,
                excitations=self.Nexcitations,
            )

        self.size = len(self.basis)

    def default_waveguide(self):
        raise Exception("default_waveguide() method undefined")

    def basis_state(self, state: State) -> NDArray[np.double]:
        """Create the wavefunction for a pure state with the given occupations.

        Args:
            state (tuple[int,...]): sorted tuple with occupied modes

        Returns:
            wavefunction (NDArray[double]): wavefunction a 1 in the given position
        """
        if state not in self.basis:
            raise Exception(f"State {state} is not on our basis")
        wavefunction = np.zeros(self.size)
        wavefunction[self.basis[state]] = 1.0
        return wavefunction

    def excited_qubit(self, which=0) -> NDArray[np.double]:
        """Return the wavefunction for a state with one excitation in the given 'qubit'."""
        if which > self.Nqubits or which < 0:
            raise Exception(f"Requested qubit {which} outside the available range")
        return self.basis_state((self.qubit_indices[which],))

    def waveguide_photon(self, wavefunction: ArrayLike) -> NDArray:
        """Create the wavefunction of a photon in the waveguide.

        Args:
            wavefunction (ArrayLike): amplitudes of the wavefunction modes
        Returns:
            wavefunction (NDArray): wavefunction of the waveguide, cavities and qubits
        """
        if len(wavefunction) != self.waveguide.modes:
            raise Exception("Photon wavefunction does not match number of modes")
        wavefunction = np.asarray(wavefunction)
        state = np.zeros(self.size, dtype=wavefunction.dtype)
        for n, psi_n in zip(self.waveguide_indices, wavefunction):
            state[self.basis[(n,)]] = psi_n
        return state

    def mode_occupations(self, wavefunction: ArrayLike) -> NDArray[np.double]:
        """Extract the weights of the bosonic modes from a 1D or 2D array of
        wavefunctions."""
        return mode_occupations(self.basis, wavefunction)


@dataclass
class Exp_2qubits_2cavities_2purcells(Setup):
    """Class describing an experiment with two qubits, placed in two cavities, at the
    sides of a quantum link, with the connection mediated by Purcell filters.

    The class admits the following parameters:

    δ1:        qubit 1 gap
    δ2:        qubit 2 gap
    g1:        qubit 1 - cavity 1 coupling (see below)
    g2:        qubit 2 - cavity 2 coupling
    gp1:       qubit 1 - filter 1 coupling
    gp2:       qubit 2 - filter 2 coupling
    ω1:        cavity 1 frequency
    ω2:        cavity 2 frequency
    ωp1:       Purcell filter 1 frequency
    ωp2:       Purcell filter 2 frequency
    κ1:        Purcell filter 1 decay
    κ2:        Purcell filter 2 decay
    δLamb:     Computed Lamb shifts for the qubits

    The couplings g1 and g2 may be either constants, or functions g1(t) and g2(t)
    that return the values of time-dependent controls.
    """

    Nqubits: int = 2
    Ncavities: int = 2
    Nfilters: int = 2
    Nexcitations: int = 1

    δ1: float = 2 * π * 8.406
    δ2: float = 2 * π * 8.406
    g1: Union[Number, Callable] = 2 * π * 0.0086
    g2: Union[Number, Callable] = 2 * π * 0.0086
    gp1: float = 2 * π * 0.025
    gp2: float = 2 * π * 0.025
    ω1: float = 2 * π * 8.406
    ω2: float = 2 * π * 8.406
    ωp1: float = 2 * π * 8.406
    ωp2: float = 2 * π * 8.406
    κ1: float = 2 * π * 0.0086
    κ2: float = 2 * π * 0.0086
    δLamb: float = 0.0

    H: sp.csr_matrix = field(init=False)
    H_qb1_cav1: sp.csr_matrix = field(init=False)
    H_qb2_cav2: sp.csr_matrix = field(init=False)

    def default_waveguide(self):
        return Waveguide(frequency=self.ω1 + self.δLamb, length=5, modes=351)

    def __post_init__(self):

        super().__post_init__()

        """ Construct the vector of energies E for filling in the diagonal entries. """

        self.E = np.array(
            [
                self.δ1 + self.δLamb,
                self.δ2 + self.δLamb,
                self.ω1,
                self.ω2,
                self.ωp1,
                self.ωp2,
            ]
            + self.waveguide.frequencies.tolist()
        )

        H_diag = diagonals_with_energies(self.basis, self.E)

        """Fill in the entries for cavity- waveguide couplings (in this case purcell-waveguide)"""

        H_purcell_waveguide_x0 = sum(
            G
            * move_excitation_operator(
                origin_mode=self.filter_indices[0],
                destination_mode=self.waveguide_indices[k],
                basis=self.basis,
            )
            for k, G in enumerate(self.waveguide.coupling_strength(self.ω1, self.κ1, 0))
        )
        H_purcell_waveguide_xl = sum(
            G
            * move_excitation_operator(
                origin_mode=self.filter_indices[1],
                destination_mode=self.waveguide_indices[k],
                basis=self.basis,
            )
            for k, G in enumerate(
                self.waveguide.coupling_strength(
                    self.ω2, self.κ2, self.waveguide.length
                )
            )
        )

        H_cavity_purcell = sum(
            [
                g
                * move_excitation_operator(
                    origin_mode=self.cavity_indices[i],
                    destination_mode=self.filter_indices[i],
                    basis=self.basis,
                )
                for i, g in enumerate([self.gp1, self.gp2])
            ]
        )

        """ Up untill here this is the whole static part"""

        H_off_diag = H_cavity_purcell + H_purcell_waveguide_x0 + H_purcell_waveguide_xl

        self.H = H_diag + H_off_diag + H_off_diag.H

        """ Here we initialize the dynamical part, but only for later use in Hamiltonian() """

        self.H_qb1_cav1 = move_excitation_operator(
            origin_mode=self.qubit_indices[0],
            destination_mode=self.cavity_indices[0],
            basis=self.basis,
        )
        self.H_qb2_cav2 = move_excitation_operator(
            origin_mode=self.qubit_indices[1],
            destination_mode=self.cavity_indices[1],
            basis=self.basis,
        )

    def Hamiltonian(self, t=0.0):
        """Compute the Hamiltonian at time 't'"""
        g1 = self.g1
        if not np.isscalar(g1):
            g1 = g1(t)
        g2 = self.g2
        if not np.isscalar(g2):
            g2 = g2(t)
        Ht = g1 * self.H_qb1_cav1 + g2 * self.H_qb2_cav2
        return self.H + Ht + Ht.H


class Exp_2qubits_2cavities:
    """Class describing an experiment with two qubits, placed in two cavities, at the
    sides of a quantum link.

    The class admits the following parameters:

    δ1:        qubit 1 gap
    δ2:        qubit 2 gap
    g1:        qubit 1 - cavity 1 coupling (see below)
    g2:        qubit 2 - cavity 2 coupling
    ω1:        cavity 1 frequency
    ω2:        cavity 2 frequency
    κ1:        cavity 1 decay
    κ2:        cavity 2 decay
    δLamb:     Computed Lamb shifts for the qubits

    The couplings g1 and g2 may be either constants, or functions g1(t) and g2(t)
    that return the values of time-dependent controls.
    """

    Nqubits: int = 2
    Ncavities: int = 2
    Nfilters: int = 0
    Nexcitations: int = 1

    δ1: float = 2 * π * 8.406
    δ2: float = 2 * π * 8.406
    g1: Union[Number, Callable] = 2 * π * 0.0086
    g2: Union[Number, Callable] = 2 * π * 0.0086
    ω1: float = 2 * π * 8.406
    ω2: float = 2 * π * 8.406
    κ1: float = 2 * π * 0.0086
    κ2: float = 2 * π * 0.0086
    δLamb: float = 0.0

    H: sp.csr_matrix = field(init=False)
    H_qb1_cav1: sp.csr_matrix = field(init=False)
    H_qb2_cav2: sp.csr_matrix = field(init=False)

    def default_cavity(self):
        return Waveguide(frequency=self.ω1 + self.δLamb, length=5, modes=351)

    def __post_init__(self):

        super().__post_init__()

        """ Construct the vector of energies E for filling in the diagonal entries. """

        self.E = np.array(
            [
                self.δ1 + self.δLamb,
                self.δ2 + self.δLamb,
                self.ω1,
                self.ω2,
            ]
            + self.waveguide.frequencies.tolist()
        )

        H_diag = diagonals_with_energies(self.basis, self.E)

        """Fill in the entries for cavity- waveguide couplings (in this case purcell-waveguide)"""

        H_cavity_waveguide_x0 = sum(
            G
            * move_excitation_operator(
                origin_mode=self.cavity_indices[0],
                destination_mode=self.waveguide_indices[k],
                basis=self.basis,
            )
            for k, G in enumerate(self.waveguide.coupling_strength(self.ω1, self.κ1, 0))
        )
        H_cavity_waveguide_xl = sum(
            G
            * move_excitation_operator(
                origin_mode=self.cavity_indices[1],
                destination_mode=self.waveguide_indices[k],
                basis=self.basis,
            )
            for k, G in enumerate(
                self.waveguide.coupling_strength(
                    self.ω2, self.κ2, self.waveguide.length
                )
            )
        )

        """ Up untill here this is the whole static part"""

        H_off_diag = H_cavity_waveguide_x0 + H_cavity_waveguide_xl

        self.H = H_diag + H_off_diag + H_off_diag.H

        """ Here we initialize the dynamical part, but only for later use in Hamiltonian() """

        self.H_qb1_cav1 = move_excitation_operator(
            origin_mode=self.qubit_indices[0],
            destination_mode=self.cavity_indices[0],
            basis=self.basis,
        )
        self.H_qb2_cav2 = move_excitation_operator(
            origin_mode=self.qubit_indices[1],
            destination_mode=self.cavity_indices[1],
            basis=self.basis,
        )

    def Hamiltonian(self, t=0.0):
        """Compute the Hamiltonian at time 't'"""
        g1 = self.g1
        if not np.isscalar(g1):
            g1 = g1(t)
        g2 = self.g2
        if not np.isscalar(g2):
            g2 = g2(t)
        Ht = g1 * self.H_qb1_cav1 + g2 * self.H_qb2_cav2
        return self.H + Ht + Ht.H
