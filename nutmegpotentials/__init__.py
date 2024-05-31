import openmmml
from .nutmegpotential import NutmegPotentialImplFactory

openmmml.mlpotential.MLPotential.registerImplFactory('nutmeg-small', NutmegPotentialImplFactory())
openmmml.mlpotential.MLPotential.registerImplFactory('nutmeg-medium', NutmegPotentialImplFactory())
openmmml.mlpotential.MLPotential.registerImplFactory('nutmeg-large', NutmegPotentialImplFactory())

