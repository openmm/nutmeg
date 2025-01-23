# Nutmeg Potentials

This repository contains the Nutmeg machine learning potentials described in

Peter Eastman, Benjamin P. Pritchard, John D. Chodera, Thomas E. Markland.  "Nutmeg and SPICE: Models and Data for
Biomolecular Machine Learning."  J. Chem. Theory Comput. 20, 19, 8583-8593 (2024).  https://doi.org/10.1021/acs.jctc.4c00794

They are made available in several formats.

1. Pytorch models in TorchScript format
2. A potential function for use with [OpenMM-ML](https://github.com/openmm/openmm-ml)
3. A Calculator for use with [ASE](https://wiki.fysik.dtu.dk/ase/index.html)
4. The source code and checkpoints for the original PhysicsML models

Instructions for using the first three are given below.  The PhysicsML models are a less
convenient form, and there is usually no reason to use them directly.  They can be found
in the `source` directory.  They are not installed with the package.

## Installation

1. Download this repository.
2. In a terminal, `cd` to the top level directory containing `setup.py`.
3. Enter the command

```
pip install .
```

## Usage: TorchScript models

The Pytorch models in TorchScript format are in the directory `nutmegpotentials/models`.
You can load them with `torch.jit.load()`.

```python
import torch
model = torch.jit.load('nutmeg-small.pt')
```

Each model requires the following information as inputs.

- The atomic positions in nm.
- The elements of the atoms.
- The Gasteiger partial charges of the atoms, as computed with RDKit.  Other types of
  partial charges will probably not produce accurate results and should not be used.
- The periodic box vectors in nm, if periodic boundary conditions are to be applied.

The following example uses RDKit to construct an alanine molecule from a SMILES string and
compute the necessary information.

```python
from rdkit import Chem
from rdkit.Chem import rdPartialCharges, rdDistGeom
mol = Chem.MolFromSmiles('C[CH](N)C(O)=O')
mol = Chem.AddHs(mol)
rdPartialCharges.ComputeGasteigerCharges(mol)
rdDistGeom.EmbedMultipleConfs(mol, numConfs=1)
positions = 0.1*mol.GetConformer(0).GetPositions()
symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
charges = [a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()]
```

The elements and charges need to be passed to the model in a particular pre-digested form.
You can use the `create_atom_features()` function to create the necessary input tensors.
The positions and box vectors (if present) must also be passed as Pytorch tensors.

```python
from nutmegpotentials import create_atom_features
types, node_attrs = create_atom_features(symbols, charges)
positions = torch.tensor(positions, dtype=torch.float32)
```

You can now invoke the model.

```python
energy = model(positions, types, node_attrs, None)
```

The return value is the energy in kJ/mol.  We have passed `None` for the box vectors so
that periodic boundary conditions will not be applied.  Alternatively we could pass a
(3, 3) tensor with the vectors defining the periodic box.

## Usage: OpenMM

This package includes a potential function for use with OpenMM-ML.  Simply specify
the name of the model to use.

```python
import nutmegpotentials
from openmmml import MLPotential
potential = MLPotential('nutmeg-small')
```

You can then pass a `Topology` object to it from which to create a `System`.  In addition,
it needs Gasteiger partial charges for the atoms.  One option is to pass an array of charges
to `createSystem()`.

```python
system = potential.createSystem(topology, charges=charges)
```

As an alternative, it can use RDKit to automatically determine the partial charges.  In
that case you only need to provide the total charge of the system as an integer.

```python
system = potential.createSystem(topology, total_charge=0)
```

You can also create mixed systems in which part is modelled with a Nutmeg model and part
with a conventional force field.  See the OpenMM-ML documentation for details.

## Usage: ASE

This package includes a Calculator for use with ASE.  To create it, you specify the
name of the model to use, the `Atoms` object it will be used to simulate, the Gasteiger
partial charges of the atoms, and the Pytorch device to perform computations on.  The
`Atoms` object must include atomic symbols.

```python
import ase
import torch
from nutmegpotentials.nutmegcalculator import NutmegCalculator
atoms = ase.Atoms(symbols=symbols, positions=positions)
device = torch.device('cuda')
atoms.calc = NutmegCalculator('nutmeg-small', atoms, charges, device)
```

When setting positions for the `Atoms` object, remember that ASE measures distances
in Angstroms.  The Calculator automatically performs conversions between the units
used by ASE and the ones used internally by the models.
