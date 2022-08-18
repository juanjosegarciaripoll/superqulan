from ast import Call
import numpy as np
import scipy.sparse.linalg
from typing import Callable, Optional, Union
from numpy.typing import NDArray, ArrayLike


def Trotter_solver_dynamics(
    times: ArrayLike,
    v0: NDArray,
    H: Callable[[float, NDArray], NDArray],
    collect: Optional[Union[bool, Callable]] = lambda t, v0: v0,
):
    """Integrate a time-dependent Schrödinger equation, with initial condition
    and given integration time-steps.

    The routine assumes that the user provides a vector of monotonously growing
    time steps, times[i] <= times[i+1]. The evolution among time-steps is
    estimated assuming that the Hamiltonian is approximately constant during
    those times.

    If 'collect' is provided and is a function, 'collect(t, state)' is invoked
    and the output is collected at each of these time steps. By default,
    the routine collects all wavefunctions. However, if 'collect' is 'None'
    the simulator only returns the final state.

    Args:
        times (ArrayLike): sequence of times use for the integration
        v0 (NDArray): initial state wavefunction
        H (Callable): function H(t) returning the Hamiltonian at time 't'
        collect (Callable or None): strategy for collecting information

    Returns:
        output (NDArray): If 'collect' is False, a 1D array containing the final
        state. Otherwise, it contains a 2D array, the rows of which are the output
        of 'collect(t,state)' or the 'state' evaluated at times[i].
    """

    output = []
    last_t = times[0]
    for step, t in enumerate(times):
        if step > 0:
            dt = t - last_t
            v0 = scipy.sparse.linalg.expm_multiply((-1j * dt) * H(last_t + dt / 2), v0)
        if collect:
            output.append(collect(times[0], v0))
        last_t = t

    if collect:
        return np.array(output).T
    else:
        return np.array(v0)


def Stochastic_solver_dynamics(
    times: ArrayLike,
    state: ArrayLike,
    H: Callable,
    rates: ArrayLike[float],
    jumps: list,
    collect: Callable = lambda t, state: state,
):
    """Integrate a master equation using stochastic trajectories.

    The routine assumes that the user provides a vector of monotonously growing
    time steps, times[i] <= times[i+1]. The evolution among time-steps is
    estimated assuming that the Hamiltonian is approximately constant during
    those times.

    The user must also provide a set of decay 'rates' and 'jumps' operators
    associated to the master equation.

    If 'collect' is provided and is a function, 'collect(t, state)' is invoked
    and the output is collected at each of these time steps. By default,
    the routine collects all wavefunctions.

    Args:
        times (ArrayLike): sequence of times used for the integration
        state (NDArray): initial state wavefunction
        H (Callable): function H(t) returning the Hamiltonian at time 't'
        collect (Callable): strategy for collecting information
        rates (ArrayLike): list of decay rates for the master equation
        jumps (list): list of jump operators associated to the master equation

    Returns:
        output (NDArray): If 'collect' is False, a 1D array containing the final
        state. Otherwise, it contains a 2D array, the rows of which are the output
        of 'collect(t,state)' or the 'state' evaluated at times[i].
    """

    output = []
    last_t = times[0]
    projectors = list(jump_i.H @ jump_i for jump_i in jumps)
    imagH = sum(-0.5j * rate_i * P_i for rate_i, P_i in zip(rates, projectors))
    last_t = times[0]

    for step, t in enumerate(times):
        if step > 0:
            dt = t - last_t
            jumped = False
            for gamma_i, P_i, jump_i in zip(rates, projectors, jumps):
                P_jump = dt * gamma_i * np.vdot(state, P_i @ state)
                if np.random.rand() <= P_jump:
                    state = jump_i @ state
                    state /= np.linalg.norm(state)
                    jumped = True
                    break
            if not jumped:
                state = scipy.sparse.linalg.expm_multiply(
                    (-1j * dt) * (H(last_t + dt / 2) + imagH), state
                )
                state /= np.linalg.norm(state)
        output.append(collect(times[0], state))
        last_t = t

    return np.array(output).T
