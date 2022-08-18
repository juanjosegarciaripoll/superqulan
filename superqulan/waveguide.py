from math import pi as π
import numpy as np


class Waveguide:
    def __init__(
        self,
        frequency: float,
        length: float = 5,
        modes: int = 100,
        linearize: bool = False,
    ):
        """Create a realistic representation of a WR90 waveguide of 'length', for a
        subset of 'nmax' modes centered around 'frequency'. If 'linearize' is True
        the dispersion relation will be linearized around that frequency. Otherwise
        it will follow that of the W90 waveguide.

        Args:
            frequency (float): Central frequency (2π x GHz) for the selected modes
            length (float, optional): Waveguide length in meters
            modes (int, optional): Number of modes. Defaults to 100.
            linearize (bool, optional): Replace the actual dispersion relation by linear
        """
        mode_indices = np.arange(length * 80)

        self.waveguide_width = 0.02286
        self.speed_of_light = 299792458
        self.length = length

        mode_frequencies = (
            (2 * π)
            * (
                self.speed_of_light
                * np.sqrt(
                    (1 / (2 * self.waveguide_width)) ** 2
                    + (mode_indices / (2 * length)) ** 2
                )
            )
            / 1e9
        )

        self.modes = modes

        central_index = np.abs(mode_frequencies - frequency).argmin()
        central_frequency = mode_frequencies[central_index]
        self.central_index = central_index
        self.lowest_index = max(central_index - int(self.modes / 2), 0)
        self.highest_index = min(self.lowest_index + self.modes, len(mode_indices))

        self.linearized = linearize
        self.linearized_group_velocity = None

        if linearize:
            self.linearized_group_velocity = self.group_velocity(central_frequency)
            domega = (
                mode_frequencies[central_index + 1]
                - mode_frequencies[central_index - 1]
            ) / 2
            mode_frequencies = central_frequency + domega * (
                mode_indices - central_index
            )

        self.frequencies = mode_frequencies[self.lowest_index : self.highest_index]
        # super().__init__(sp.diags(self.frequencies, 0))  #This we'll make differently
        self.tprop = (self.length / (self.group_velocity(central_frequency))) * 1e9

    def group_velocity(self, frequency: float) -> float:
        """Return the group velocity in m/s for the given 'frequency' in 2π x GHz."""
        if self.linearized_group_velocity is not None:
            return self.linearized_group_velocity
        else:
            frequency_nu = frequency * 1e9 / (2 * π)
            return self.speed_of_light * np.sqrt(
                1
                - (self.speed_of_light / (2 * self.waveguide_width * frequency_nu)) ** 2
            )

    def coupling_strength(
        self, frequency: float, kappa: float, position: float
    ) -> float:
        """Compute the coupling strength for an oscillator of given 'frequency'
        that is coupled to this environment and decays with rate 'kappa' (in
        GHz). This is the factor that will multiply the self.coupling_at()
        vector."""
        return (
            np.sqrt(
                ((kappa * 1e9) * self.group_velocity(frequency))
                / (2 * self.length)
                / frequency
            )
            * self.mode_weight(position)
            / 1e9
        )

    def mode_weight(self, position: float) -> np.ndarray:
        """Return the vector of couplings for a quantum object connected at
        given 'position' to the waveguide.

        Args:
            position (float): location of the coupled object in meters

        Returns:
            np.ndarray: vector of couplings to the different modes
        """
        k = np.arange(self.lowest_index, self.highest_index) / self.length
        mode_wavefunction = np.cos(π * position * k)
        return np.sqrt(self.frequencies) * mode_wavefunction
