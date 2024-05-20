from utils.VQA import VQA
import logging
import os
from PIL import Image
from tqdm import tqdm
import json
from utils.DDP_manager import DDP
import torch
import torch.multiprocessing as mp
from utils.datasets import Image_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from utils.arg_parse import argparse_captioning

class DDP_Captioning(DDP):
    def __init__(
        self, 
        rank, 
        world_size, 
        dataset_path,
        images,
        captions
    ):
        self.images = images
        self.dataset_path = dataset_path
        self.captions = captions
        super(DDP_Captioning, self).__init__(rank, world_size)
    
    def print(self, *args, **kwargs):
        print(f'Rank {self.rank}:', *args, **kwargs)

    def main(self):
        captioner = VQA(
            device = self.device,
            opt = {
                'vqa_model': ("Llava", "utils/llava/weights/llava-v1.5-13b"),
                'logger': logging.getLogger(),
            }
        )
        question = 'Can you describe the image?'

        dt = Image_dataset(self.images, self.dataset_path)
        loader = DataLoader(
            dt,
            batch_size=None,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=DistributedSampler(dt, shuffle=False),
        )

        for image, image_name in tqdm(loader, desc=f'Rank {self.rank}', position=self.rank):
            image = captioner.process_image(image)
            answer = captioner.get_caption(image, question)
            self.captions[image_name] = answer

def run(rank, opt, world_size, images, captions):
    torch.manual_seed(opt['seed'])
    DDP_Captioning(rank, world_size, opt['dataset_path'], images, captions)

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    opt = argparse_captioning()
    dataset_path = opt['dataset_path']

    manager = mp.Manager()
    captions = manager.dict()

    images = os.listdir(dataset_path)

    mp.spawn(run, args=(
                    opt,
                    world_size, 
                    images,
                    captions
                ), nprocs=world_size)
    
    with open(os.path.join(dataset_path, 'captions.json'), 'w') as f:
        json.dump(dict(captions), f, indent=4)