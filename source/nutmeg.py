import torch
import zbl_tensornet
from physicsml.lightning.graph_datasets.neighbourhood_list_torch import construct_edge_indices_and_attrs
from typing import Optional

class Nutmeg(torch.nn.Module):
    """This class acts as a wrapper around the PhysicsML model, providing a
    simpler interface for invoking it."""

    def __init__(self, model, cutoff, self_interaction):
        super(Nutmeg, self).__init__()
        self.model = model
        self.cutoff = cutoff
        self.self_interaction = self_interaction

    def forward(self, positions, types, node_attrs, boxvectors: Optional[torch.Tensor] = None):
        """Execute the model and compute the potential energy.
        
        Parameters
        ----------
        positions: torch.Tensor
            Tensor of shape (N, 3) containing the atom positions in nm
        types: torch.Tensor
            Tensor of length N containing the atom type index of each atom
        node_attrs: torch.Tensor
            Tensor of length (N, 18) containing the input node features for
            each atom.  The first 17 elements are the one-hot encoded atom
            type, and the final element is the partial charge.
        boxvectors: torch.Tensor
            Tensor of shape (3, 3) containing the periodic box vectors.  If
            this is None, periodic boundary conditions are not used.
        
        Returns
        -------
        the potential energy in kJ/mol
        """
        positions = positions.to(torch.float32)
        if boxvectors is None:
            cell = torch.eye(3, dtype=torch.float32)
            pbc = (False, False, False)
        else:
            cell = boxvectors.to(torch.float32)
            pbc = (True, True, True)
        edge_index, edge_attrs, cell_shift_vector = construct_edge_indices_and_attrs(
                positions=positions, cutoff=self.cutoff, initial_edge_indices=None, initial_edge_attrs=None,
                pbc=pbc, cell=cell, self_interaction=self.self_interaction)
        data = {
            'coordinates': positions,
            'edge_index': edge_index,
            'cell_shift_vector': cell_shift_vector,
            'raw_atomic_numbers': types,
            'node_attrs': node_attrs,
            'num_nodes': torch.tensor(positions.shape[0], device=positions.device),
            'batch': torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device),
            'num_graphs': torch.tensor(1, device=positions.device)
        }
        data['pbc'] = torch.tensor(pbc, device=positions.device)
        data['cell'] = cell.to(positions.device)
        if edge_attrs is not None:
            data['edge_attrs'] = edge_attrs
        result = self.model(data)
        return result['y_graph_scalars']

