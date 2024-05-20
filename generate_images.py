import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from utils.datasets import Proposed_biases
from utils.DDP_manager import DDP
import utils.arg_parse as arg_parse
from torch.utils.data import Dataset

# required to run eval()
from utils.generative_models import Stable_Diffusion_XL, Stable_Diffusion

# MULTI DATA PARALLELIZATION
import torch.multiprocessing as mp

# split the dataset into chunks for each rank
class Distributed_dataset(Dataset):
    def __init__(self, 
            rank, 
            world_size, 
            opt,
            ds
        ):
        self.rank = rank
        self.world_size = world_size
        # get data
        data = ds.get_data()
        self.data_to_generate = []
        for prompt, caption_id in data:
            # if the folder does not exist, add it to the list of data to generate
            if not os.path.isdir(os.path.join(opt['save_path'], str(caption_id))):
                self.data_to_generate.append((prompt, caption_id))
            # if the folder exists, check if the number of images is less than the desired number of images
            else:
                length = len(os.listdir(os.path.join(opt['save_path'], str(caption_id))))
                if length < opt['dataset_setting']['n-images']:
                    # if the number of images is less than the desired number of images, add it to the list of data to generate
                    # NOTE: this will overwrite the existing images
                    self.data_to_generate.append((prompt, caption_id))

        # split data
        length = len(self.data_to_generate)
        samples_per_rank = length // world_size
        if rank == world_size-1:
            self.data_to_generate = self.data_to_generate[rank*samples_per_rank:]
        else:
            self.data_to_generate = self.data_to_generate[rank*samples_per_rank: (rank+1)*samples_per_rank]

    def __getitem__(self, idx):
        caption, caption_id = self.data_to_generate[idx]
        return caption, caption_id
    
    def __len__(self):
        return len(self.data_to_generate)

class DDP_image_gen(DDP):
    def __init__(
        self, 
        rank, 
        world_size,
        opt,
        ds
    ):
        self.seed = opt['seed']
        self.gen_info = opt['generator']
        self.save_path = opt['save_path']
        os.makedirs(self.save_path, exist_ok=True)
        self.n_images = opt['dataset_setting']['n-images']
        self.batch_size = opt['gen_setting']['batch_size']
        self.pos_prompt = self.gen_info['pos_prompt']
        self.opt = opt
        self.ds = ds
        super(DDP_image_gen, self).__init__(rank, world_size)
    
    def split_batches(self, l, n_images):
        for i in range(0, len(l), n_images):
            yield l[i: i+n_images]

    def main(self):
        # init generative model
        generative_model = eval(self.gen_info['class'])(gen_info = self.gen_info, device = self.device, n_images=self.n_images)
        # get dataset for the specific rank
        ds = Distributed_dataset(
            rank = self.rank,
            world_size = self.world_size,
            opt = self.opt,
            ds = self.ds
        )
        
        print(f'Rank {self.rank} has {len(ds)} samples to generate')
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        # generate and save images
        for prompts, caption_ids in tqdm(loader, position=self.rank, desc=f'Rank {self.rank}'):
            torch.cuda.empty_cache()
            prompts = [p+' '+self.pos_prompt for p in prompts]
            gen_images = generative_model.generate_images(prompt=prompts)
            batch_images = self.split_batches(gen_images, self.n_images)
            for batch_idx, images in enumerate(batch_images):
                caption_id = str(caption_ids[batch_idx].item()) if type(caption_ids[batch_idx]) == torch.Tensor else str(caption_ids[batch_idx])
                save_dir = os.path.join(self.save_path, caption_id)
                os.makedirs(save_dir, exist_ok=True)
                # for each generated image in the batch
                for image_idx, image in enumerate(images):
                    # save image
                    image.save(os.path.join(save_dir, f'{image_idx}.jpg'))
                    # check if image was saved correctly
                    if not os.path.isfile(os.path.join(save_dir, f'{image_idx}.jpg')):
                        print(f'ERROR: image {image_idx} of caption {caption_id} not saved')

def run(
    rank, 
    world_size,
    opt,
    ds
):  
    # Set seed
    torch.manual_seed(opt['seed'])
    DDP_image_gen(
        rank = rank,
        world_size = world_size,
        opt = opt,
        ds = ds
    )

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f'Using {world_size} GPUs')
    # Parse arguments
    opt = arg_parse.argparse_generate_images()
    mp.set_start_method('spawn')

    # Load dataset
    ds = Proposed_biases(
        dataset_path = opt['dataset_setting']['proposed_biases_path'],
        max_prompts = opt['gen_setting']['max_prompts_per_bias'],
        filter_threshold = opt['gen_setting']['filter_threshold'],
        hard_threshold = opt['gen_setting']['hard_threshold'],
        merge_threshold = opt['gen_setting']['merge_threshold'],
        valid_bias_fn = opt['dataset_setting']['valid_bias_fn'],
        filter_caption_fn = opt['dataset_setting']['filter_caption_fn'],
        all_images = opt['dataset_setting']['all_images']
    )
    # Start DDP
    mp.spawn(
        run, 
        args=(world_size, opt, ds, ), 
        nprocs=world_size
    )