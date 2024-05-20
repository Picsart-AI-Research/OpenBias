from utils.llama_wrapper import Llama_2
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils.utils as utils
import json
import torch

'''
    This class is used to propose biases given a set of prompts (ideally from a dataset)
    Workflow:
    1. in-context learning is performed on a LLM (Llama2) to propose biases that may be present on a target generative model
    2. Check whether the response is in valid JSON format
    3. If yes, it is considered a bias proposal

    This base class is extended for the dataset specific version (e.g., COCO, Flickr)
'''
class Bias_Proposal_Base():
    def __init__(
        self,
        opt
    ):
        # Init Llama2
        self.LLM = Llama_2(
            ckpt_dir=opt['llama2']['weights_path'],
            tokenizer_path=opt['llama2']['tokenizer_path'],
            max_seq_len=opt['max_seq_len'],
            max_batch_size=opt['batch_size'],
            model_parallel_size=opt['model_parallel_size'],
            rank = opt['rank'],
            max_gen_len = opt['max_gen_len'],
            temperature = opt['temperature'],
            top_p = opt['top_p'],
            seed = opt['seed'],
            SYSTEM_PROMPT = opt['system_prompt']
        )
        self.rank = opt['rank']
        dataset = opt['dataset_setting']['dataset'](opt)
        opt['logger'].info(f"Dataset size: {len(dataset)}")
        # Create dataloader
        self.loader = DataLoader(
            dataset, 
            batch_size=opt['batch_size'], 
            shuffle=False, 
            num_workers=opt['workers'], 
            pin_memory=True, 
            sampler=DistributedSampler(dataset, shuffle=False)
        )
    
    # Propose biases given a set of sentences
    def propose_biases(self, sentences):
        LLM_reply = self.LLM.generate(sentences)

        proposed_biases = []
        for idx, answer in enumerate(LLM_reply):
            # answer from LLM
            ans = answer['generation']['content'].replace("\n", "")
            # parse the answer:
            # get the string after the first {
            if '{' in ans:
                ans = '{'+ans.split('{', 1)[1]
            # get the string until the last }
            ans = ans.rsplit('}', 1)[0]+'}'
            # check whether the answer is in valid JSON format
            if utils.is_json(ans):
                proposed_biases.append((True, json.loads(ans)))
            else:
                proposed_biases.append((False, ans))
        return proposed_biases

# This class is used to propose biases for the COCO dataset
class Bias_proposal_coco(Bias_Proposal_Base):
    def __init__(self, opt):
        super(Bias_proposal_coco, self).__init__(opt)
    
    def run_proposals(self):
        biases = []
        not_proposed_biases = []
        for captions, image_ids, caption_ids in tqdm(self.loader, position=self.rank, desc=f"Rank {self.rank}"):
            proposed_biases = self.propose_biases(captions)
            for caption, image_id, caption_id, proposed_bias in zip(captions, image_ids, caption_ids, proposed_biases):
                if proposed_bias[0]:
                    biases.append(
                        {
                            "image_id": image_id.item() if type(image_id) == torch.Tensor else image_id,
                            "caption_id": caption_id.item() if type(caption_id) == torch.Tensor else caption_id,
                            "caption": caption,
                            "proposed_biases": proposed_bias[1]
                        }
                    )
                else:
                    not_proposed_biases.append(
                        {
                            "image_id": image_id.item() if type(image_id) == torch.Tensor else image_id,
                            "caption_id": caption_id.item() if type(caption_id) == torch.Tensor else caption_id,
                            "caption": caption,
                            "LLM_answer": proposed_bias[1]
                        }
                    )
        return biases, not_proposed_biases

# This class is used to propose biases for the Flickr dataset
class Bias_proposal_flickr(Bias_Proposal_Base):
    def __init__(self, opt):
        super(Bias_proposal_flickr, self).__init__(opt)
    
    def run_proposals(self):
        biases = []
        not_proposed_biases = []
        for captions, image_ids in tqdm(self.loader, position=self.rank, desc=f"Rank {self.rank}"):
            proposed_biases = self.propose_biases(captions)
            for caption, image_id, proposed_bias in zip(captions, image_ids, proposed_biases):
                if proposed_bias[0]:
                    biases.append(
                        {
                            "image_id": image_id,
                            "caption_id": image_id.split('.')[0],
                            "caption": caption,
                            "proposed_biases": proposed_bias[1]
                        }
                    )
                else:
                    not_proposed_biases.append(
                        {
                            "image_id": image_id,
                            "caption_id": image_id.split('.')[0],
                            "caption": caption,
                            "LLM_answer": proposed_bias[1]
                        }
                    )
        return biases, not_proposed_biases