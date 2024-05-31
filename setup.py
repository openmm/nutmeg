from setuptools import setup

setup(
    name='nutmeg',
    author='Peter Eastman',
    description='The Nutmeg machine learning potentials',
    version='1.0',
    license='MIT',
    url='https://github.com/openmm/nutmeg',
    packages=['nutmegpotentials'],
    entry_points={
        'openmmml.potentials': [
            'nutmeg-small = nutmegpotentials.nutmegpotential:NutmegPotentialImplFactory',
            'nutmeg-medium = nutmegpotentials.nutmegpotential:NutmegPotentialImplFactory',
            'nutmeg-large = nutmegpotentials.nutmegpotential:NutmegPotentialImplFactory'
        ]
    }
)


