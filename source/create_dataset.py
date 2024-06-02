import numpy as np
from openff.toolkit.topology import Molecule
import openff
from openmm.unit import *
from collections import defaultdict
from datasets import Dataset
from molflux.datasets import split_dataset
from molflux.splits import load_from_dict
import h5py

typeDict = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Si': 9, 'P': 10, 'S': 11, 'Cl': 12, 'K': 13, 'Ca': 14, 'Br': 15, 'I': 16}

posScale = 0.0529177210903 #1*bohr/nanometer
energyScale = 2625.499639479826 #1*hartree/item/(kilojoules_per_mole)
forceScale = energyScale/posScale

def sample_generator():
    infile = h5py.File('SPICE-2.0.1.hdf5')
    for name in infile:
        g = infile[name]
        count = len(g['atomic_numbers'])
        molSmiles = g['smiles'].asstr()[0]
        mol = Molecule.from_mapped_smiles(molSmiles, allow_undefined_stereo=True)
        molTypes = [typeDict[atom.symbol] for atom in mol.atoms]
        assert len(molTypes) == count
        for i, atom in enumerate(mol.atoms):
            assert atom.atomic_number == g['atomic_numbers'][i]
        mol.assign_partial_charges('gasteiger')
        q = [[a.partial_charge.m] for a in mol.atoms]
        if not np.all(np.isfinite(q)):
            print(molSmiles)
            continue
        numConfs = g['conformations'].shape[0]
        for i in range(numConfs):
            yield {
                'smiles': molSmiles,
                'physicsml_coordinates': np.array(g['conformations'][i], dtype=np.float32)*posScale,
                'physicsml_atom_numbers': np.array(molTypes, dtype=np.int32),
                'physicsml_atom_features': np.array(q, dtype=np.float32),
                'formation_energy': np.float32(g['formation_energy'][i]*energyScale),
                'forces': -np.array(g['dft_total_gradient'][i], dtype=np.float32)*forceScale
            }

dataset = Dataset.from_generator(sample_generator)

config = {
          'name': 'shuffle_split',
          'presets':
            {
              'train_fraction': 0.95,
              'validation_fraction': 0.05,
              'test_fraction': 0.0,
            }
          }

strategy = load_from_dict(config)
datasets = next(split_dataset(dataset, strategy))
datasets.save_to_disk('physicsml_dataset')

