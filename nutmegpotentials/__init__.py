import openmmml
from .nutmegpotential import NutmegPotentialImplFactory
from .util import create_atom_features

openmmml.mlpotential.MLPotential.registerImplFactory('nutmeg-small', NutmegPotentialImplFactory())
openmmml.mlpotential.MLPotential.registerImplFactory('nutmeg-medium', NutmegPotentialImplFactory())
openmmml.mlpotential.MLPotential.registerImplFactory('nutmeg-large', NutmegPotentialImplFactory())

