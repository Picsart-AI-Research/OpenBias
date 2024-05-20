from argparse import RawTextHelpFormatter
import argparse
from utils.config import VQA_SETTING, BIAS_PROPOSAL_SETTING, GEN_SETTING, LOGGING_CONFIG, VQA_EVALUATION
import logging
import logging.config
import torch
import modelscope, os

# bias proposal arg parser
def argparse_bias_proposals():
    parser = argparse.ArgumentParser(description='Bias Proposals', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dataset', choices=['coco', 'flickr_30k', 'ffhq', 'stylegan3_ffhq', 'winobias'], help="dataset to use")
    opt = vars(parser.parse_args())
    opt['dataset_setting'] = BIAS_PROPOSAL_SETTING[opt['dataset']]
    opt['max_seq_len'] = BIAS_PROPOSAL_SETTING['max_seq_len']
    opt['seed'] = BIAS_PROPOSAL_SETTING['seed']
    opt['batch_size'] = BIAS_PROPOSAL_SETTING['batch_size']
    opt['system_prompt'] = opt['dataset_setting']['system_prompt']
    opt['llama2'] = BIAS_PROPOSAL_SETTING['llama2']
    opt['save_path'] = f'proposed_biases/{opt["dataset"]}/{opt["dataset_setting"]["n_prompts_per_image"]}'
    opt['json_path'] = f'{opt["save_path"]}/{opt["dataset"]}{opt["dataset_setting"]["mode"]}.json'
    opt['not_json_path'] = f'{opt["save_path"]}/{opt["dataset"]}{opt["dataset_setting"]["mode"]}_not_json.json'
    logging.config.dictConfig(LOGGING_CONFIG)
    opt['logger'] = logging.getLogger('bias_proposal')

    # llama2 args
    opt['model_parallel_size'] = 1
    opt['temperature'] = 0.6
    opt['top_p'] = 0.9
    opt['max_gen_len'] = None
    return opt

# image generation arg parser
def argparse_generate_images():
    parser = argparse.ArgumentParser(description='Image Generation', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dataset', choices=['coco', 'flickr_30k', 'winobias'], help="dataset to use")
    parser.add_argument('--generator', choices=list(GEN_SETTING['generators'].keys()), help="generator to use")
    opt = vars(parser.parse_args())
    opt['dataset_setting'] = GEN_SETTING[opt['dataset']]
    opt['gen_setting'] = GEN_SETTING
    opt['seed'] = GEN_SETTING['seed']
    opt['save_path'] = os.path.join(
        opt['gen_setting']['save_path'],
        opt['dataset_setting']['subfolder'],
        opt['generator'],
    )
    opt['generator'] = GEN_SETTING['generators'][opt['generator']]
    logging.config.dictConfig(LOGGING_CONFIG)
    opt['logger'] = logging.getLogger('image_generation')

    return opt

# VQA arg parser
def argparse_VQA():
    parser = argparse.ArgumentParser(description='VQA', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dataset', choices=['coco', 'flickr_30k', 'ffhq', 'winobias', 'stylegan3_ffhq'], help="dataset to use")
    parser.add_argument('--vqa_model', choices=list(VQA_SETTING['vqa_models'].keys()), help="vqa model to use")
    parser.add_argument('--mode', choices=['original', 'generated'], help="use original or generated images")
    parser.add_argument('--generator', choices=list(GEN_SETTING['generators'].keys()), help="Generated images to run the VQA on")
    opt = vars(parser.parse_args())
    if opt['mode'] == 'generated' and opt['generator'] is None:
        raise ValueError("Please specify a generator to use with --generator")

    opt['dataset_setting'] = VQA_SETTING[opt['dataset']][opt['mode']]
    if opt['mode'] == 'generated':
        opt['dataset_setting']['images_path'] = os.path.join(
            GEN_SETTING['save_path'],
            opt['dataset_setting']['subfolder'],
            opt['generator'],
            opt['dataset_setting']['inner_folder']
        )
    
    opt['valid_bias_fn'] = opt['dataset_setting']['valid_bias_fn']
    opt['max_prompts_per_bias'] = opt['dataset_setting']['max_prompts_per_bias']
    opt['UNK_CLASS'] = VQA_SETTING['UNK_CLASS']
    opt['seed'] = VQA_SETTING['seed']
    opt['save_path'] = VQA_SETTING['save_path']
    opt['filter_threshold'] = GEN_SETTING['filter_threshold']
    opt['hard_threshold'] = GEN_SETTING['hard_threshold']
    opt['merge_threshold'] = GEN_SETTING['merge_threshold']
    opt['vqa_model_name'] = opt['vqa_model']
    opt['vqa_model'] = VQA_SETTING['vqa_models'][opt['vqa_model']]

    modelscope.get_logger().disabled = True
    opt['logger'] = logging.getLogger('VQA')
    # loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    # quit()
    return opt

# VQA evaluation arg parser
def argparse_VQA_evaluation():
    parser = argparse.ArgumentParser(description='VQA Eval', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--vqa_model', type=str, help="vqa model to use")
    parser.add_argument('--bias_cluster', type=str, help="bias group name")
    parser.add_argument('--bias_name', type=str, help="bias name")
    parser.add_argument('--bias_att', type=str, help="bias attribute")
    parser.add_argument('--version', type=str, help="FairFace version -> race version (4 classes or 7)")
    parser.add_argument('--dataset', choices=['coco', 'flickr_30k'], help="dataset to use")
    parser.add_argument('--mode', choices=['original', 'generated'], help="use original or generated images")
    parser.add_argument('--generator', choices=list(GEN_SETTING['generators'].keys()), help="Generated images to run the VQA on")
    opt = vars(parser.parse_args())

    if opt['mode'] == 'generated' and opt['generator'] is None:
        raise ValueError("Please specify a generator to use with --generator")

    opt['vqa_model_name'] = opt['vqa_model']
    opt['vqa_model'] = VQA_EVALUATION['vqa_models'][opt['vqa_model']]
    opt['prediction_mapper'] = VQA_EVALUATION['mapper']
    opt['dataset_setting'] = VQA_SETTING[opt['dataset']][opt['mode']]
    if opt['mode'] == 'generated':
        opt['dataset_setting']['images_path'] = os.path.join(
            GEN_SETTING['save_path'],
            opt['dataset_setting']['subfolder'],
            opt['generator'],
        )
    opt['max_prompts'] = VQA_EVALUATION['max_prompts']
    opt['filter_threshold'] = VQA_EVALUATION['filter_threshold']
    opt['hard_threshold'] = VQA_EVALUATION['hard_threshold']
    opt['merge_threshold'] = VQA_EVALUATION['merge_threshold']
    opt['results_path'] = VQA_EVALUATION['results_path']
    opt['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    modelscope.get_logger().disabled = True
    logger = modelscope.get_logger()
    opt['logger'] = logging.getLogger('VQA_eval')
    # loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    # quit()
    return opt

# StyleGAN3 arg parser
def argparse_stylegan3():
    parser = argparse.ArgumentParser(description='StyleGAN3', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--n_images', type=int)
    parser.add_argument('--generator', choices=['stylegan3-ffhq'], help="generator to use")
    opt = vars(parser.parse_args())

    opt['seed'] = GEN_SETTING['seed']
    opt['save_path'] = os.path.join(
        GEN_SETTING['save_path'],
        opt['generator'],
    )
    opt['generator'] = GEN_SETTING['generators'][opt['generator']]
    os.makedirs(opt['save_path'], exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
    opt['logger'] = logging.getLogger('image_generation')
    
    return opt

# captioning arg parser
def argparse_captioning():
    parser = argparse.ArgumentParser(description='Captioning', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--generator', choices=['stylegan3-ffhq'], help="dataset to use")
    opt = vars(parser.parse_args())
    opt['seed'] = GEN_SETTING['seed']
    opt['dataset_path'] = os.path.join(GEN_SETTING['save_path'], opt['generator'])
    logging.config.dictConfig(LOGGING_CONFIG)
    opt['logger'] = logging.getLogger('captioning')
    return opt

def argparse_fair_face_eval():
    parser = argparse.ArgumentParser(description='FairFace eval', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dataset', choices=['coco', 'flickr_30k', 'ffhq'], help="dataset to use")
    parser.add_argument('--mode', choices=['original', 'generated'], help="use original or generated images")
    # parser.add_argument('--full_dataset', action='store_true', help="use full dataset")
    parser.add_argument('--generator', choices=list(GEN_SETTING['generators'].keys()), help="Generated images to run the VQA on")
    opt = vars(parser.parse_args())

    opt['dataset_setting'] = VQA_SETTING[opt['dataset']][opt['mode']]
    opt['save_path'] = VQA_SETTING['save_path']

    if opt['mode'] == 'generated':
        opt['dataset_setting']['images_path'] = os.path.join(
            GEN_SETTING['save_path'],
            opt['dataset_setting']['subfolder'],
            opt['generator'],
            opt['dataset_setting']['inner_folder']
        )
    opt['max_prompts'] = opt['dataset_setting']['max_prompts_per_bias']
    opt['filter_threshold'] = VQA_SETTING['filter_threshold']
    opt['hard_threshold'] = VQA_SETTING['hard_threshold']
    opt['merge_threshold'] = VQA_SETTING['merge_threshold']
    
    opt['valid_bias_fn'] = opt['dataset_setting']['valid_bias_fn']

    opt['vqa_model'] = VQA_SETTING['vqa_models']['llava-1.5-13b']
    
    modelscope.get_logger().disabled = True
    opt['logger'] = logging.getLogger('VQA')
    
    return opt