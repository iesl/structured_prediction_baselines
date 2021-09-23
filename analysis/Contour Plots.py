# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os, sys

path = os.path.dirname(os.path.abspath('.'))
if not path in sys.path: sys.path.append(path)

# %%
import torch
from allennlp.models import Model
from allennlp.common import Params
from allennlp.common import plugins
from allennlp.common import util as common_util
common_util.import_module_and_submodules('structured_prediction_baselines')

# %%
from allennlp.data import (
    Field,
    DataLoader,
    DatasetReader,
    Instance,
    Batch,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.training import Trainer
from allennlp.nn import util

from structured_prediction_baselines.common import ModelMode

# %% [markdown]
# # Load model, dataset-reader, data and vocab

# %%
serialization_dir = "../model_wts/"  # path to the directory containing config, model weights and vocab
config_file = os.path.join(serialization_dir, "config.json")
vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
weights_file = os.path.join(serialization_dir, "weights.th")

# %%
loaded_params = Params.from_file(config_file)
loaded_model = Model.load(loaded_params, serialization_dir, weights_file)
loaded_model.eval()

dataset_reader = DatasetReader.from_params(loaded_params.get("dataset_reader"))

data_loader_params = loaded_params.get("data_loader")
train_data_loader = DataLoader.from_params(
    params=data_loader_params.duplicate(),
    reader=dataset_reader,
    data_path='../' + loaded_params.get("train_data_path")
)

dev_data_loader = DataLoader.from_params(
    reader=dataset_reader,
    data_path='../' + loaded_params.get("validation_data_path"),
    params=data_loader_params,
)

vocab = Vocabulary.from_instances(train_data_loader.iter_instances())

# %%
train_data_loader.index_with(vocab)
dev_data_loader.index_with(vocab)

# %%
# uncomment if you want to load trainer
# loaded_params['trainer']['cuda_device'] = -1
# loaded_params['trainer']['num_epochs'] = 1

# trainer = Trainer.from_params(
#     model=loaded_model,
#     serialization_dir=serialization_dir,
#     data_loader=train_data_loader,
#     validation_data_loader=dev_data_loader,
#     params=loaded_params.pop("trainer"),
# )

# %% [markdown]
# ## Get an instance from the data

# %%
train_instances = train_data_loader.iter_instances()
inst0 = next(train_instances)
instances = [inst0]

# %%
loaded_model.forward_on_instance(inst0, mode=ModelMode.COMPUTE_SCORE)

# %%
device = 0 if torch.cuda.is_available() else -1
with torch.no_grad():
    dataset = Batch(instances)
    dataset.index_instances(vocab)
    model_input = util.move_to_device(dataset.as_tensor_dict(), device=device)

# %%
model_input

# %%
with torch.no_grad():
    model_output = loaded_model(model_input['x'], model_input['labels'], mode=ModelMode.COMPUTE_SCORE)

# %%
model_output

# %% [markdown]
# # Contour Plots

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

# %%
y0 = np.linspace(0, 1, 500)
y1 = np.linspace(0, 1, 500)
z = np.zeros((500, 500))

# %%
x = model_input['x']
y = model_input['labels'].clone().detach().float()
for i in range(len(y0)):
    for j in range(len(y1)):
        y[0][0] = y0[i]
        y[0][1] = y1[j]
        with torch.no_grad():
            z[i][j] = loaded_model(x, y, mode=ModelMode.COMPUTE_SCORE)['score']

# %%
plt.contour(y1, y0, z, 20, colors='black', )
plt.colorbar()
plt.xlabel("y1")
plt.ylabel("y0")
plt.show()

# %%
