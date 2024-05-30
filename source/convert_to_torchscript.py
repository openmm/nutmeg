import torch
from nutmeg import Nutmeg
from molflux.core import load_model

def convert_model(model_path, output_path):
    model = load_model(model_path)
    del model.module.losses
    del model.module.model_config
    module = torch.jit.script(Nutmeg(model.module.to_torchscript(),
                                     model.config['datamodule']['cut_off'],
                                     model.config['datamodule']['self_interaction']))
    module.save(output_path)

convert_model('nutmeg-small', 'nutmeg-small.pt')
convert_model('nutmeg-medium', 'nutmeg-medium.pt')
convert_model('nutmeg-large', 'nutmeg-large.pt')

