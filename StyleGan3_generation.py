from utils import generative_models
import torch
from tqdm import tqdm
import os
import utils.arg_parse as arg_parse
from PIL import Image
import torch.multiprocessing as mp
from utils.DDP_manager import DDP

class DDP_image_gen(DDP):
    def __init__(self, rank, world_size, opt):
        self.opt = opt
        super().__init__(rank, world_size)
    
    def main(self):
        n_images = self.opt['n_images']
        gan_model = generative_models.StyleGAN3(
            device = self.device,
            gen_info = self.opt['generator'],
        )
        # seeds for this rank
        seeds = [i for i in range(self.rank, n_images, self.world_size)]
        for seed in tqdm(seeds, desc=f'Rank {self.rank}', position=self.rank):
            img = gan_model.generate_images(seed)
            img = Image.fromarray(img.numpy(), 'RGB')
            os.makedirs(os.path.join(self.opt['save_path'], str(seed)), exist_ok=True)
            img.save(os.path.join(self.opt['save_path'], str(seed), '0.png'))

def run(rank, world_size, opt):
    torch.manual_seed(opt['seed'])
    DDP_image_gen(
        rank = rank,
        world_size = world_size,
        opt = opt
    )

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f'Using {world_size} GPUs')
    opt = arg_parse.argparse_stylegan3()
    mp.spawn(
        run, 
        args=(world_size, opt,), 
        nprocs=world_size
    )