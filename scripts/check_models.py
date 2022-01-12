import argparse
import torch
from collections import OrderedDict

device='cpu'

# model_path_a="/mnt/nfs/scratch1/purujitgoyal/structured_prediction_baselines/xtropy_model_weights/update/cal500_best.th"
# model_path_b=".allennlp_models/run-20210903_192540-3tccxzny/model_state_e30_b0.th"
# model_path_a="/mnt/nfs/scratch1/purujitgoyal/structured_prediction_baselines/.allennlp_models/run-20210901_015139-2s3twc81/best.th"

model_path_a="xtropy_model_weights/update/eurlexev_best_jy.th"
model_path_b="xtropy_model_weights/update/eurlexev_best.th"

model_state_dict1 = torch.load(model_path_a, map_location=torch.device(device))
model_state_dict2 = torch.load(model_path_b, map_location=torch.device(device))

key_list=['sampler.inference_nn.feature_network._linear_layers.0.weight', 'sampler.inference_nn.feature_network._linear_layers.0.bias', 'sampler.inference_nn.feature_network._linear_layers.1.weight', 'sampler.inference_nn.feature_network._linear_layers.1.bias', 'sampler.inference_nn.feature_network._linear_layers.2.weight', 'sampler.inference_nn.feature_network._linear_layers.2.bias', 'sampler.inference_nn.feature_network._linear_layers.3.weight', 'sampler.inference_nn.feature_network._linear_layers.3.bias', 'sampler.inference_nn.feature_network._linear_layers.4.weight', 'sampler.inference_nn.feature_network._linear_layers.4.bias', 'sampler.inference_nn.label_embeddings.weight']


for k1,v1 in model_state_dict1.items():
    if k1 in key_list:
        v2=model_state_dict2[k1]
        assert torch.all(torch.eq(v1,v2))


## purujit's code for converting the keys.
# import argparse
# import torch
# from collections import OrderedDict
# ​
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ​
# parser = argparse.ArgumentParser()
# parser.add_argument('--load_model_path', type=str, help='path to the model', required=True)
# parser.add_argument('--save_model_path', type=str, help='path to save the updated model', required=True)
# ​
# args = parser.parse_args()
# ​
# model_state_dict = torch.load(args.load_model_path, map_location=torch.device(device))
# updated_state_dict = OrderedDict([
# 		(k.replace('sampler.constituent_samplers.0', 'sampler.tasknn'), v) 
# 			if k.startswith('sampler.constituent_samplers.0') 
# 			else (k, v) for k, v in model_state_dict.items()
# 	])
# ​
# torch.save(updated_state_dict, args.save_model_path)
