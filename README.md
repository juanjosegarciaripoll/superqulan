# Quantum link modelization

This library, developed within the Horizon 2020 FET-Open project SuperQuLAN [www.superqulan.eu](www.superqulan.eu), offers tools to simulate the dynamics of two small superconducting quantum processors connected by a microwave quantum link.

This includes:
1. Efficient representation of the states and operators with one or more excitations, in the Hilbert space formed by the various qubits, cavities and waveguides.
2. Representations of specific setups, such as those by the SuperQuLAN consortium.
3. Tools to simulate the dynamics of those setups.
4. Examples of applications of those tools to analyze state transfer, implementation of gates, etc.

## Structure

The project is composed of the following elements:
- `superqulan` is the folder that contains the library components;
- `examples` is a folder containing notebooks that use the library to simulate state transfer and other quantum operations;
- `setup.py`, `requirements.txt`, etc, are files that enable the installation of the library as a Python package.

The library itself is composed of the following submodules:
- `bosons.py` contains the routines that build up a Hilbert space basis together with the most important operators, the information required is the number of fermionic modes, the number of bosonic modes and the amount of excitations.
- `waveguide.py` constructs the object that joins the different quantum nodes. One can specify the length as well as the number of modes.
- `simulator.py` contains the function to implement time evolution.
- `architecture.py` makes use of bosons.py and waveguide.py to create full distributed quantum architectures consisting of nodes and links.

## Usage

This is a pure Python package that may be installed from the root directory as
```bash
pip install .
```
However, this step is not required to run the notebooks in the `examples` directory.

## To be done

- Incorporate other waveguide representations
- Write down examples working with two or more excitations
- Implement simulations with qubit / waveguide decay using stochastic trajectories
- Add documentation

## Acknowledgments and references

This library has been used to model the implementation of distributed quantum gates and state transfer in two linked quantum computers, by [Guillermo F. Peñas et al, Phys. Rev. Appl. 17, 054038 (2021)](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.17.054038). Cite this work or the most up-to-date Zotero reference for this repository if you use or extend this software.