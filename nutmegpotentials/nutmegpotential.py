from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory
import openmm
import os
from typing import Iterable, Optional

class NutmegPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates ANIPotentialImpl objects."""

    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return NutmegPotentialImpl(name)


class NutmegPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the Nutmeg models."""

    def __init__(self, name):
        self.name = name

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  includeBonds: bool = True,
                  **args):
        import torch
        import openmmtorch
        from .util import create_atom_features
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        if 'charges' in args:
            charges = args['charges']
        else:
            from rdkit import Chem
            from rdkit.Chem import rdDetermineBonds, rdPartialCharges
            rdmol = Chem.EditableMol(Chem.Mol())
            for atom in topology.atoms():
                a = Chem.Atom(atom.element.atomic_number)
                a.SetNoImplicit(True)
                rdmol.AddAtom(a)
            for bond in topology.bonds():
                rdmol.AddBond(bond[0].index, bond[1].index, Chem.BondType.SINGLE)
            rdmol = rdmol.GetMol()
            Chem.SanitizeMol(rdmol)
            rdDetermineBonds.DetermineBondOrders(rdmol, args['total_charge'], embedChiral=False)
            rdPartialCharges.ComputeGasteigerCharges(rdmol)
            charges = [rdmol.GetAtomWithIdx(atom.index).GetDoubleProp('_GasteigerCharge') for atom in includedAtoms]
        symbols = [atom.element.symbol for atom in includedAtoms]
        types, node_attrs = create_atom_features(symbols, charges)
        model = torch.jit.load(os.path.join(os.path.dirname(__file__), 'models', f'{self.name}.pt'))

        class NutmegForce(torch.nn.Module):

            def __init__(self, model, types, node_attrs, atoms):
                super(NutmegForce, self).__init__()
                self.model = model
                self.types = torch.nn.Parameter(types, requires_grad=False)
                self.node_attrs = torch.nn.Parameter(node_attrs, requires_grad=False)
                if atoms is None:
                    self.indices = None
                else:
                    self.indices = torch.tensor(sorted(atoms), dtype=torch.int64)

            def forward(self, positions: torch.Tensor, boxvectors: Optional[torch.Tensor] = None):
                if self.indices is not None:
                    positions = positions[self.indices]
                return self.model.forward(positions, self.types, self.node_attrs, boxvectors)

        # Create the TorchForce and add it to the System.

        module = torch.jit.script(NutmegForce(model, types, node_attrs, atoms))
        force = openmmtorch.TorchForce(module)
        force.setForceGroup(forceGroup)
        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)

        # Add dummy bonds so OpenMM won't break up molecules when applying
        # periodic boundary conditions and barostats.

        if includeBonds:
            bonds = openmm.CustomBondForce('0')
            bonds.setName('NutmegDummyBonds')
            bonds.setForceGroup(forceGroup)
            system.addForce(bonds)
            atomSet = set(includedAtoms)
            for a1, a2 in topology.bonds():
                if a1 in atomSet and a2 in atomSet:
                    bonds.addBond(a1.index, a2.index, [])

