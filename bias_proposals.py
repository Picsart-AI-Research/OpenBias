import utils.arg_parse as arg_parse
import json
import torch
import torch.multiprocessing as mp
from utils.DDP_manager import DDP
import os

class DDP_bias_proposal(DDP):
    def __init__(
        self, 
        rank, 
        world_size,
        proposed_biases = [],
        no_biases_proposed = [],
        opt = {}
    ):
        self.proposed_biases = proposed_biases
        self.no_biases_proposed = no_biases_proposed
        self.opt = opt
        super(DDP_bias_proposal, self).__init__(rank, world_size)
        
    def main(
        self        
    ):
        self.opt['rank'] = self.rank
        bias_proposal = self.opt['dataset_setting']['bias_proposal_module'](self.opt)
        biases, not_proposed_biases = bias_proposal.run_proposals()
        self.proposed_biases += biases
        self.no_biases_proposed += not_proposed_biases

def run(
    rank: int, 
    world_size: int,
    proposed_biases = [],
    no_biases_proposed = [],
    opt = {}
):
    DDP_bias_proposal(
        rank, 
        world_size,
        proposed_biases,
        no_biases_proposed,
        opt
    )

def main(opt):
    # Set seed 
    torch.manual_seed(opt['seed'])
    # Initialize MULTI GPUs
    opt['logger'].info(f"Initialize MULTI GPUs on {torch.cuda.device_count()} devices")
    world_size = torch.cuda.device_count()
    # Initialize manager for shared memory
    manager = mp.Manager()

    # Initialize shared memory
    proposed_biases = manager.list()
    no_biases_proposed = manager.list()

    mp.spawn(run, args=(world_size, 
                        proposed_biases,
                        no_biases_proposed,
                        opt
                    ), nprocs=world_size)

    proposed_biases = json.dumps({"bias_proposal": list(proposed_biases)}, indent=4)
    no_bias_proposed = json.dumps({"no_bias_proposed": list(no_biases_proposed)}, indent=4)
    os.makedirs(opt['save_path'], exist_ok=True)
    # Saving outputs
    with open(opt['json_path'], "w+") as outfile:
        outfile.write(proposed_biases)
    with open(opt['not_json_path'], "w+") as outfile:
        outfile.write(no_bias_proposed)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    # Parse bias proposal arguments
    opt = arg_parse.argparse_bias_proposals()
    # Run main
    main(opt)