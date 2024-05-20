import json
from copy import deepcopy

with open('1/stylegan3_ffhq.json', 'r') as f:
    biases = json.load(f)['bias_proposal']

for info in biases:
    info['caption_id'] = int(info['image_id'].split('.')[0])

with open('1/stylegan3_ffhq_with_cpt_ids.json', 'w') as f:
    json.dump({'bias_proposal': biases}, f, indent=4)

