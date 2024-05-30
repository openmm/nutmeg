"""
This file defines the PhysicsML model class used by Nutmeg.  It extends the
standard TensorNet implementation (PooledTensorNetModule) and adds a repulsive
ZBL potential.
"""

from typing import Any, Dict, List, Type
from pydantic.dataclasses import dataclass
from physicsml.lightning.config import ConfigDict
from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.tensor_net.supervised.default_configs import TensorNetModelConfig
from physicsml.models.tensor_net.supervised.tensor_net_module import PooledTensorNetModule
from physicsml.models.utils import compute_lengths_and_vectors
from molflux.modelzoo import register_model
from molflux.modelzoo.info import ModelInfo
from dataclasses import field
from torch_geometric.utils import scatter
import torch

@dataclass(config=ConfigDict)
class ZBLTensorNetModelConfig(TensorNetModelConfig):
    atomic_number_map: List[int] = field(default_factory=lambda: [])

@register_model(kind="physicsml", name="zbl_tensor_net_model")
class ZBLTensorNetModel(PhysicsMLModelBase[ZBLTensorNetModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[ZBLTensorNetModelConfig]:
        return ZBLTensorNetModelConfig

    def _instantiate_module(self) -> Any:
        return ZBLTensorNetModelModule(model_config=self.model_config)

class ZBLTensorNetModelModule(PooledTensorNetModule):
    model_config: ZBLTensorNetModelConfig

    def __init__(
        self,
        model_config: ZBLTensorNetModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config=model_config)
        atomic_number_map = torch.as_tensor(model_config.atomic_number_map, dtype=torch.int)
        self.register_buffer("atomic_number_map", atomic_number_map)
        # This maps atomic numbers to covalent radii.  The values are from https://doi.org/10.1063/1.1725697.
        radii = {1: 0.025, 3: 0.145, 4: 0.105, 5: 0.085, 6: 0.07, 7: 0.065, 8: 0.06, 9: 0.05, 11: 0.18, 12: 0.15,
                 13: 0.125, 14: 0.11, 15: 0.1, 16: 0.1, 17: 0.1, 19: 0.22, 20: 0.18, 21: 0.16, 22: 0.14, 23: 0.135,
                 24: 0.14, 25: 0.14, 26: 0.14, 27: 0.135, 28: 0.135, 29: 0.135, 30: 0.135, 31: 0.13, 32: 0.125,
                 33: 0.115, 34: 0.115, 35: 0.115, 37: 0.235, 38: 0.2, 39: 0.18, 40: 0.155, 41: 0.145, 42: 0.145,
                 43: 0.135, 44: 0.13, 45: 0.135, 46: 0.14, 47: 0.16, 48: 0.155, 49: 0.155, 50: 0.145, 51: 0.145,
                 52: 0.14, 53: 0.14}
        radius_map = torch.tensor([radii[n] for n in model_config.atomic_number_map], dtype=torch.float32)
        self.register_buffer("radius_map", radius_map)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # The following is copied from PooledTensorNetModule.  I would just call super().forward(), but
        # TorchScript doesn't understand subclasses.

        data = self.embedding(data)

        for interaction in self.interactions:
            data = interaction(data)

        output = {}
        if self.scalar_output is not None:
            pooled_output: torch.Tensor = (
                self.scalar_output(data) * self.scaling_std + self.scaling_mean
            )

            if "total_atomic_energy" in data:
                pooled_output = pooled_output + data["total_atomic_energy"].unsqueeze(
                    -1,
                )

            output["y_graph_scalars"] = pooled_output

        if self.node_scalar_output is not None:
            node_output: torch.Tensor = (
                self.node_scalar_output(data) * self.scaling_std + self.scaling_mean
            )
            output["y_node_scalars"] = node_output

        # Look up the atomic numbers and covalent radii of the atoms.

        edge_index = data["edge_index"]
        atomic_number = self.atomic_number_map[data['raw_atomic_numbers'][edge_index]]
        atomic_number = torch.unsqueeze(atomic_number, 2)
        radius = self.radius_map[data['raw_atomic_numbers'][edge_index]]
        radius = torch.unsqueeze(radius, 2)

        # Compute the distances between atoms.

        if "cell" in data:
            cell = data["cell"]
            cell_shift_vector = data["cell_shift_vector"]
        else:
            cell = None
            cell_shift_vector = None
        lengths, vectors = compute_lengths_and_vectors(
            positions=data["coordinates"],
            edge_index=edge_index,
            cell=cell,
            cell_shift_vector=cell_shift_vector,
        )

        # Compute the ZBL potential. 5.29e-2 is the Bohr radius in nm.  All other numbers are magic constants from the ZBL potential.

        a = 0.8854 * 5.29177210903e-2 / (atomic_number[0] ** 0.23 + atomic_number[1] ** 0.23)
        d = lengths / a
        f = 0.1818 * torch.exp(-3.2*d) + 0.5099 * torch.exp(-0.9423*d) + 0.2802 * torch.exp(-0.4029*d) + 0.02817 * torch.exp(-0.2016*d)

        # Multiply by the cutoff function.

        phi_ji = torch.where(
            lengths < radius[0]+radius[1],
            0.5 * (torch.cos(torch.pi * lengths / (radius[0]+radius[1])) + 1),
            torch.zeros_like(lengths)
        )
        f *= phi_ji

        # Compute the energy.  The prefactor is 1/(4*pi*eps0) in kJ*nm/mol.  Multiply by 0.5 because every
        # atom pair appears twice.

        energy = f * atomic_number[0] * atomic_number[1] / lengths
        energy = 0.5 * 138.9354576 * scatter(src=energy, index=data['batch'][edge_index[0]], dim=0, dim_size=output['y_graph_scalars'].shape[0])
        output['y_graph_scalars'] = output['y_graph_scalars']+energy
        return output

    def compute_loss(self, input: Any, target: Any) -> torch.Tensor:
        return super().compute_loss(input, target)
