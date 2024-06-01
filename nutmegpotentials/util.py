def create_atom_features(symbols, charges):
    """This utility function creates the input tensors that need to be passed
    to a Nutmeg model's forward() method to describe the atoms.
    
    Parameters
    ----------
    symbols: list of str
        the element symbol for each atom in the system to be simulated
    charges: list of float
        the Gasteiger partial charge of each atom in the system to be simulated

    Returns
    -------
    types: torch.Tensor
        the type index of each atom
    node_attrs: torch.Tensor
        the feature vector for each atom
    """
    import torch
    typeDict = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Si': 9, 'P': 10, 'S': 11, 'Cl': 12, 'K': 13, 'Ca': 14, 'Br': 15, 'I': 16}
    types = torch.tensor([typeDict[symbol] for symbol in symbols], dtype=torch.int64)
    one_hot_z = torch.nn.functional.one_hot(types, num_classes=17).to(torch.float32)
    charges = torch.tensor([[c] for c in charges], dtype=torch.float32)
    node_attrs = torch.cat([one_hot_z, charges], dim=1)
    return types, node_attrs

