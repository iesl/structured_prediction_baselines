import argparse
import torch
from collections import OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--load_model_path', type=str, help='path to the model', required=True)
parser.add_argument('--save_model_path', type=str, help='path to save the updated model', required=True)

args = parser.parse_args()

model_state_dict = torch.load(args.load_model_path, map_location=torch.device(device))
updated_state_dict = OrderedDict([
		(k.replace('sampler.constituent_samplers.0', 'sampler'), v) 
			if k.startswith('sampler.constituent_samplers.0') 
			else (k, v) for k, v in model_state_dict.items()
	])

torch.save(updated_state_dict, args.save_model_path)
