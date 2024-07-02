import ase.calculators.calculator
import torch
import os
from typing import List

class NutmegCalculator(ase.calculators.calculator.Calculator):
    """This an an ASE Calculator for performing calculations with the Nutmeg models."""

    def __init__(self, modelname, atoms, charges, device):
        """
        Construct a Calculator for one of the Nutmeg models.

        Parameters
        ----------
        modelname: str
            the name of the model.  This must be either 'nutmeg-small', 'nutmeg-medium', or 'nutmeg-large'.
        atoms: ase.Atoms
            the ASE Atoms object to simulate.  It must contain atomic symbols.
        charges: array
            Gasteiger partial charges for the atoms.
        device: torch.Device
            the device on which to perform calculations
        """
        super().__init__()
        self.implemented_properties: List[str] = ["energy", "forces"]
        self.atoms = atoms
        model = torch.jit.freeze(torch.jit.load(os.path.join(os.path.dirname(__file__), 'models', f'{modelname}.pt')).eval().to(device))
        self.model = model.to(device)
        self.device = device
        from .util import create_atom_features
        types, node_attrs = create_atom_features(list(atoms.symbols), charges)
        self.types = types.to(device)
        self.node_attrs = node_attrs.to(device)
        self.pos_scale = 0.1 # convert A to nm
        self.energy_scale = 96.48533212331002 # convert eV to kJ/mol

    def calculate(self, atoms, properties, system_changes):
        positions = torch.tensor(atoms.get_positions()*self.pos_scale, dtype=torch.float32, requires_grad=True, device=self.device)
        if any(atoms.pbc):
            boxvectors = torch.tensor(atoms.cell*self.pos_scale, device=self.device)
        else:
            boxvectors = None
        with torch.jit.optimized_execution(False): # https://github.com/pytorch/pytorch/issues/69078#issuecomment-1087217720
            energy = self.model.forward(positions, self.types, self.node_attrs, boxvectors)/self.energy_scale
        self.results['energy'] = energy.detach().cpu().item()
        energy.backward()
        self.results['forces'] = -(positions.grad*self.pos_scale).detach().cpu().numpy()

