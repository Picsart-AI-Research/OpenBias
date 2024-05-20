import utils.arg_parse as arg_parse
import torch
import torch.multiprocessing as mp
from utils.DDP_manager import DDP
from utils.VQA import VQA
from utils.datasets import VQA_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image
import json
import os

class DDP_VQA(DDP):
    def __init__(
        self, 
        rank, 
        world_size,
        bias_counts,
        vqa_answers,
        opt
    ):
        self.vqa_model = opt['vqa_model']
        dataset_setting = opt['dataset_setting']
        self.proposed_biases_path = dataset_setting['proposed_biases_path']
        self.image_paths = dataset_setting['images_path']
        self.max_prompts = opt['max_prompts_per_bias']
        self.opt = opt
        self.vqa_answers = vqa_answers
        self.bias_counts = bias_counts
        super(DDP_VQA, self).__init__(rank, world_size)

    def main(self):
        # Initialize VQA model
        vqa_model = VQA(self.device, self.opt)   
  
        # Initialize dataset     
        dataset = VQA_dataset(
            dataset_setting = self.opt['dataset_setting'],
            mode = self.opt['mode'],
            max_prompts = self.max_prompts,
            filter_threshold = self.opt['filter_threshold'],
            hard_threshold = self.opt['hard_threshold'], 
            merge_threshold = self.opt['merge_threshold'],
            valid_bias_fn = self.opt['valid_bias_fn'],
            filter_caption_fn = self.opt['dataset_setting']['filter_caption_fn'],
        )
        loader = DataLoader(
            dataset, 
            batch_size=None, 
            shuffle=False, 
            num_workers=self.opt['workers'], 
            pin_memory=True, 
            sampler=DistributedSampler(dataset, shuffle=False)
        )

        # run VQA to quantify bias
        for caption_id, caption, image_id, image_path, proposed_biases in tqdm(loader, position=self.rank, desc=f'Rank {self.rank}'):
            # load image
            image = Image.open(image_path)
            answers = {}
            image = vqa_model.process_image(image)
            # for each proposed bias, run VQA
            for bias_cluster, bias_name, class_cluster, question, classes in proposed_biases:
                # add UNK class
                classes.append(self.opt['UNK_CLASS'])
                # run VQA
                answer = vqa_model.get_answer(image, question, choices=classes)
                # get VQA prediction
                class_pred = answer['multiple_choice_answer']
                # update answers
                answers[bias_name] = (
                    bias_cluster,
                    class_cluster,
                    class_pred,
                )
                # update bias counts
                self.bias_counts[bias_cluster][bias_name][class_cluster][class_pred] += 1
            # update vqa answers
            self.vqa_answers[image_path] = answers

def run(rank, world_size, bias_counts, vqa_answers, opt):
    torch.manual_seed(opt['seed'])
    DDP_VQA(rank, world_size, bias_counts, vqa_answers, opt)

# Initialize bias counts dictionary
'''
bias_counts = {
    bias_cluster: {
        bias_name: {
            class_cluster: {
                class_name: count
            }
        }
    }
}
'''
def init_bias_counts(manager, bias_classes, UNKNOWN_CLASS = 'unknown'):
    bias_counts = manager.dict()
    for bias_cluster in bias_classes:
        bias_counts[bias_cluster] = manager.dict()
        for bias_name in bias_classes[bias_cluster]:
            bias_counts[bias_cluster][bias_name] = manager.dict()
            for class_cluster in bias_classes[bias_cluster][bias_name]:
                bias_counts[bias_cluster][bias_name][class_cluster] = manager.dict()
                classes = bias_classes[bias_cluster][bias_name][class_cluster]['classes']
                for class_name in classes:
                    bias_counts[bias_cluster][bias_name][class_cluster][class_name] = 0
                bias_counts[bias_cluster][bias_name][class_cluster][UNKNOWN_CLASS] = 0
    return bias_counts

def init_answers(manager, data, opt):
    vqa_answers = manager.dict()
    if opt['mode'] == 'generated':
        for caption_id, caption, image_id, image_path, proposed_biases in data:
            vqa_answers[image_path] = manager.dict()
    elif opt['mode'] == 'original':
        for image_id, image_path, proposed_biases in data:
            vqa_answers[image_path] = manager.dict()
    return vqa_answers

def deserialize_answers(vqa_answers):
    vqa_answers = dict(vqa_answers.copy())
    for caption_id in vqa_answers:
        vqa_answers[caption_id] = dict(vqa_answers[caption_id].copy())
    return vqa_answers

def deserialize_dict(bias_counts):
    bias_counts = dict(bias_counts.copy())
    for bias_cluster in bias_counts:
        bias_counts[bias_cluster] = dict(bias_counts[bias_cluster].copy())
        for bias_name in bias_counts[bias_cluster]:
            bias_counts[bias_cluster][bias_name] = dict(bias_counts[bias_cluster][bias_name].copy())
            for class_cluster in bias_counts[bias_cluster][bias_name]:
                bias_counts[bias_cluster][bias_name][class_cluster] = dict(bias_counts[bias_cluster][bias_name][class_cluster].copy())
    return bias_counts
    
def main(opt):
    opt['logger'].info(f"Initialize MULTI GPUs on {torch.cuda.device_count()} devices")
    world_size = torch.cuda.device_count()
    manager = mp.Manager()

    # Initialize dataset     
    dataset = VQA_dataset(
        dataset_setting = opt['dataset_setting'],
        mode = opt['mode'],
        max_prompts = opt['max_prompts_per_bias'],
        filter_threshold = opt['filter_threshold'],
        hard_threshold = opt['hard_threshold'],
        merge_threshold = opt['merge_threshold'],
        valid_bias_fn = opt['valid_bias_fn'],
        filter_caption_fn = opt['dataset_setting']['filter_caption_fn'],
    )

    # Initialize bias counts dictionary shared across processes (GPUs)
    bias_counts = init_bias_counts(
        manager, 
        dataset.get_bias_classes(), 
        UNKNOWN_CLASS = opt['UNK_CLASS']
    )

    vqa_answers = init_answers(manager, dataset.get_data(), opt)

    mp.spawn(run, args=(
                        world_size, 
                        bias_counts,
                        vqa_answers,
                        opt
                    ), nprocs=world_size)

    bias_counts = deserialize_dict(bias_counts)

    vqa_answers = deserialize_answers(vqa_answers)

    # save bias counts
    counts = json.dumps(bias_counts, indent=4)
    if opt['mode'] == 'generated':
        save_path = os.path.join(
            opt['save_path'],
            opt['dataset'],
            opt['mode'],
            opt['generator'],
            opt['vqa_model_name'],
        )
    else:
        save_path = os.path.join(
            opt['save_path'],
            opt['dataset'],
            opt['mode'],
            opt['vqa_model_name'],
        )
    os.makedirs(save_path, exist_ok=True)

    file_name = 'data_counts.json'

    with open(os.path.join(save_path, file_name), 'w+') as f:
        f.write(counts)

    # save VQA answers
    answers = json.dumps(vqa_answers, indent=4)
    file_name = 'vqa_answers.json'
    with open(os.path.join(save_path, file_name), 'w+') as f:
        f.write(answers)

if __name__ == '__main__':
    opt = arg_parse.argparse_VQA()
    main(opt)
    