# Nutmeg Sources

This directory contains the source code and checkpoint files for the Nutmeg models.
In most cases there is no reason to use any of them directly.  The `nutmegpotentials`
Python module contains everything you need in a more convenient form.  They are
provided here only for reference.

This directory contains the following.

- `zbl-tensornet.py`.  This defines the PhysicsML model class used for the Nutmeg models.
  It subclasses the built in TensorNet implementation (`PooledTensorNetModule`) and adds
  a repulsive ZBL potential to the output.
- `nutmeg-small`, `nutmeg-medium`, and `nutmeg-large`.  These directories contain
  checkpoint files created with PhysicsML.  They can be loaded with
  `molflux.core.load_model()`.  This requires MolFlux and PhysicsML to be installed.
  The API for invoking them is somewhat complicated.
- `nutmeg.py`.  This acts as a wrapper around one of the PhysicsML models.  It provides
  a simpler API for invoking the model.  The TorchScript models found in the
  `nutmegpotentials` module are instances of this class.
- `convert_to_torchscript.py`.  This script loads in each of the checkpoints, converts
  it to TorchScript, and saves it to a file.  It was used to generate the models in
  the `nutmegpotentials` module.
- `create_dataset.py`.  This script was used to convert the SPICE dataset to the format
  needed by PhysicsML.  It also computes partial charges and generates the split between
  training and validation sets.

These files have been confirmed to work with PhysicsML 0.3.1, which was used to create
the TorchScript files.  They may also work with later versions, but that cannot be
guaranteed.
