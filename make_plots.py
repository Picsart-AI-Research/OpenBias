import json
import matplotlib.pyplot as plt
import math, os
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
from utils.config import GEN_SETTING

def entropy(x):
    eps = 1e-10
    x_smoothed = x + eps
    return round(-np.sum(x_smoothed * np.log(x_smoothed))/np.log(len(x)), 5)

def uniform(x):
    return np.ones(len(x)) / len(x)

def make_plot(title, xlabel, ylabel, scores, mode, path):
    plt.figure(figsize=(30, 15))
    label = [' '.join(list(dict.fromkeys(bias[0].split()+bias[1].split()))) for bias in scores]
    x_labels = label
    # values with space in between
    x_values = np.arange(len(x_labels))
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    group_width = 0.5
    plt.bar(
        x_values,
        [float(bias[3]) for bias in scores],
        color='#C5E898',
        alpha=0.95,
        edgecolor='#7f8c8d',
        width=group_width,
        label=mode
    )
    plt.title(title + f' - {mode}', fontsize=20)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(x_values, x_labels, rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, loc="upper left")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    # Adjust x-axis limits to remove space between the first bar and the y-axis
    plt.xlim(-group_width*2, len(scores))
    plt.savefig(path, format='png', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Commands description', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--generator', choices=list(GEN_SETTING['generators'].keys()), help="dataset to use")
    parser.add_argument('--dataset', choices=['coco', 'flickr_30k', 'winobias', 'ffhq'], help="dataset to use")
    parser.add_argument('--mode', choices=['original', 'generated'], help="dataset to use")
    parser.add_argument('--vqa_model', choices=['llava-1.5-13b'], default='llava-1.5-13b', help="dataset to use")
    opt = vars(parser.parse_args())

    dataset = opt['dataset']
    generator = opt['generator']
    mode = opt['mode']
    vqa_model = opt['vqa_model']

    UNK_CLASS = 'unknown'
    OTHER_CLASS = 'other'
    # As stated in the paper, we assume gender to be binary and remove the non-binary class as it is never predicted by the VQA model
    NON_BINRAY_CLASS = 'non-binary' 

    if mode == 'original':
        path = f'results/VQA/{dataset}/{mode}/{vqa_model}'
    else:
        path = f'results/VQA/{dataset}/{mode}/{generator}/{vqa_model}'

    # 1. compute context-free entropy for each bias
    with open(f'{path}/data_counts.json', 'r') as f:
        context_free_counts = json.load(f)

    # context free entropy
    entropy_final = {}
    # classes
    classes = {}
    for bias_cluster in context_free_counts:
        entropy_final[bias_cluster] = {}
        classes[bias_cluster] = {}
        for bias in context_free_counts[bias_cluster]:
            entropy_final[bias_cluster][bias] = {}

            # in the previous steps of the pipeline the class cluster must have been filtered
            # thus only one class cluster is expected
            assert len(list(context_free_counts[bias_cluster][bias].keys())) == 1, \
                f'len(list(context_free_counts[bias_cluster][bias].keys())) != 1, in the previous steps of the pipeline the class cluster must have been filtered ' \
                f'thus only one class cluster is expected'

            class_cluster = list(context_free_counts[bias_cluster][bias].keys())[0]

            local_classes = list(
                context_free_counts[bias_cluster][bias][class_cluster].keys()
            )

            local_classes.remove(UNK_CLASS)
            if OTHER_CLASS in local_classes:
                local_classes.remove(OTHER_CLASS)
            if NON_BINRAY_CLASS in local_classes:
                local_classes.remove(NON_BINRAY_CLASS)

            # get counts
            pred_counts = np.array(
                [context_free_counts[bias_cluster][bias][class_cluster][c] for c in local_classes]
            )

            # if all counts are 0, skip
            if np.sum(pred_counts) == 0:
                print(f'All counts are zeros, skipping {bias_cluster}, {bias}, {class_cluster}')
                print('This may due to the fact that the VQA always answered with the unknown class')
                continue

            # normalize
            pred_counts = pred_counts / np.sum(pred_counts)
            uniform_dist = uniform(local_classes)

            # compute and save entropy
            entropy_final[bias_cluster][bias][class_cluster] = entropy(pred_counts)

            # save classes
            classes[bias_cluster][bias] = local_classes

    # 2. compute context aware entropy for each bias
    with open(f'{path}/vqa_answers.json', 'r') as f:
        context_aware_answers = json.load(f)

    image_answers = {}
    for image in context_aware_answers:
        caption_id, image_name = image.split('/')[-2:]
        image_biases = context_aware_answers[image]
        if caption_id not in image_answers:
            image_answers[caption_id] = {}
        for vqa_bias_name in image_biases:
            cluster_name, _, vqa_cls = image_biases[vqa_bias_name]
            if cluster_name not in image_answers[caption_id]:
                image_answers[caption_id][cluster_name] = {}
            if vqa_cls != UNK_CLASS and vqa_cls != OTHER_CLASS and vqa_cls != NON_BINRAY_CLASS:
                all_classes = classes[cluster_name][vqa_bias_name]
                if vqa_bias_name not in image_answers[caption_id][cluster_name]:
                    # 0 to each class
                    image_answers[caption_id][cluster_name][vqa_bias_name] = {c: 0 for c in all_classes}
                # add 1 to the class
                image_answers[caption_id][cluster_name][vqa_bias_name][vqa_cls] += 1

    entropies_context_aware = {}
    for caption_id in image_answers:
        for bias_cluster in image_answers[caption_id]:
            if bias_cluster not in entropies_context_aware:
                entropies_context_aware[bias_cluster] = {}
            for vqa_bias_name in image_answers[caption_id][bias_cluster]:
                if vqa_bias_name not in entropies_context_aware[bias_cluster]:
                    entropies_context_aware[bias_cluster][vqa_bias_name] = []
                image_classes = np.array(list(image_answers[caption_id][bias_cluster][vqa_bias_name].values()))
                image_classes = np.array(image_classes / np.sum(image_classes))

                h = entropy(image_classes)

                if math.isnan(h) or math.isinf(h):
                    print('Entropy is nan or inf', )
                    continue
                entropies_context_aware[bias_cluster][vqa_bias_name].append(h)

    scores_entropy_context_aware = []
    scores_entropy_context_free = []
    for bias_cluster in entropy_final:
        for bias_name in entropy_final[bias_cluster]:
            name = bias_name
            # rename child to child race if bias_name == race for better visualization
            if bias_name == 'race' and bias_cluster == 'child':
                name = 'child race'
                
            for class_cluster in entropy_final[bias_cluster][bias_name]:
                h = entropy_final[bias_cluster][bias_name][class_cluster]
                entropies_context_aware_score = np.mean(entropies_context_aware[bias_cluster][bias_name])
                if not math.isnan(h) and not math.isinf(h):
                    scores_entropy_context_aware.append((bias_cluster, name, class_cluster, round(1-entropies_context_aware_score, 4)))
                    scores_entropy_context_free.append((bias_cluster, name, class_cluster, round(1-h, 4)))
                
    # sort by entropy
    scores_entropy_context_aware = sorted(scores_entropy_context_aware, key=lambda x: x[3], reverse=False)
    scores_entropy_context_free = sorted(scores_entropy_context_free, key=lambda x: x[3], reverse=False)

    # plots
    if mode != 'original':
        make_plot(
            title=dataset,
            xlabel='Bias',
            ylabel='Bias Intensity',
            scores=scores_entropy_context_aware,
            mode='context aware',
            path=os.path.join(path, 'context_aware.png')
        )

    make_plot(
        title=dataset,
        xlabel='Bias',
        ylabel='Bias Intensity',
        scores=scores_entropy_context_free,
        mode='context free',
        path=os.path.join(path, 'context_free.png')
    )