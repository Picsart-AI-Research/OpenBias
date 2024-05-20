from PIL import Image
import json
import sklearn.metrics as metrics
from collections import defaultdict
from copy import deepcopy
import json
import requests
import inflect
import os
from tqdm import tqdm
import re 

def is_json(value):
    try:
        json.loads(value)
    except Exception as e:
        return False
    return True

def extract_json_from_string(input_string):
    pattern = r'\{.*?\}'  
    match = re.search(pattern, input_string)
    
    if match:
        return match.group(0)
    else:
        return None

def extract_json(sentence):
    json_extracted = ''
    start_extraction = False
    for letter in sentence:
        if letter == "{":
            start_extraction = True
        if start_extraction:
            json_extracted += letter
            if letter == "}":
                start_extraction = False
    if len(json_extracted) != 0:
        return json_extracted
    else:
        return sentence
    
def valid_bias_generated_images(bias):
    return 'question' in bias and \
            'classes' in bias and \
            'refer_to' in bias and \
            'present_in_prompt' in bias and \
            not bias['present_in_prompt'] and \
            len(bias['classes']) > 1 and \
            isinstance(bias['refer_to'], str)

def valid_bias_real_images(bias):
    return 'question' in bias and \
            'classes' in bias and \
            'refer_to' in bias and \
            len(bias['classes']) > 1 and \
            isinstance(bias['refer_to'], str)

def valid_caption(caption):
    return  'caption_id' in caption and \
            'image_id' in caption and \
            'caption' in caption and \
            'proposed_biases' in caption and \
            'bias' in caption['proposed_biases'] and \
            isinstance(caption['proposed_biases']['bias'], list)

# needed for multiprocessing
def dd_list():
    return defaultdict(list)

def nested_dict_list():
    return defaultdict(dd_list)

def dd_str():
    return defaultdict(str)

def nested_dict_str():
    return defaultdict(dd_str)

def dd_dict():
    return defaultdict(dict)

def nested_dict():
    return defaultdict(dd_dict)

def get_max_count(cluster_classes):
    max_count = 0
    max_count_class = ''
    for c in cluster_classes:
        if cluster_classes[c]['counts'] > max_count:
            max_count = cluster_classes[c]['counts']
            max_count_class = c
    return max_count_class, max_count

def compute_overlap(list1, list2):
    setA = set(list1)
    setB = set(list2)
    overlap = setA & setB
    result1 = float(len(overlap)) / len(setA)
    result2 = float(len(overlap)) / len(setB)
    return max(result1, result2)

def get_synonyms_file():
    if os.path.isfile('utils/synonyms.json'):
        with open('utils/synonyms.json', 'r') as f:
            return json.load(f)
    return False

def get_plural_and_singular(word):
    words = [word]
    engine = inflect.engine()
    plural = engine.plural(word)
    singular = engine.singular_noun(word)
    if singular is not False:
        words.append(singular)
    words.append(plural)
    return list(set(words))

def process_word(word, type_of_relation='RelatedTo', limit=10):
    response = requests.get(f'http://api.conceptnet.io/c/en/{word}?&limit={limit}')
    try:
        edges = response.json()['edges']
    except Exception as e:
        print("ERROR")
        print(e)
        print(response)
        print(word)
        return []
    words = []
    for edge in edges:
        related_word = edge['start']['label']
        word_lan = edge['start']['language']
        weights = edge['weight']
        relation = edge['rel']['label']
        if word_lan == 'en' and word not in related_word and relation == type_of_relation:
            words.append(related_word.lower())
    return words

def filter_caption_generated(
        captions,
        bias_classes_final, 
        bias_captions_final,
    ):
    global_synonyms = {}
    generate_file = False
    # if synonyms file exists load it
    if os.path.isfile('utils/synonyms.json'):
        with open('utils/synonyms.json', 'r') as f:
            global_synonyms = json.load(f)

    words_to_check = {}
    # for each bias get the synonyms to check
    for bias_cluster in tqdm(bias_classes_final, position=0, leave=False):
        words_to_check[bias_cluster] = {}
        for bias in tqdm(bias_classes_final[bias_cluster], position=1, leave=False):
            words_to_check[bias_cluster][bias] = []
            for class_cluster in bias_classes_final[bias_cluster][bias]:
                classes = list(set(bias_classes_final[bias_cluster][bias][class_cluster]['classes']))
                for cls in classes:
                    if cls not in global_synonyms:
                        generate_file = True
                        synonyms = process_word(cls)
                        synonyms.append(cls)
                        # get plural and singular
                        p_s = []
                        for cls in synonyms:
                            p_s += get_plural_and_singular(cls)
                        synonyms += p_s
                        global_synonyms[cls] = list(set(synonyms))
                    else:
                        synonyms = global_synonyms[cls]
                    words_to_check[bias_cluster][bias] += synonyms
                words_to_check[bias_cluster][bias] = list(set(words_to_check[bias_cluster][bias]))
    
    # if file did not exist or was updated save it
    if generate_file:
        with open('utils/synonyms.json', 'w+') as f:
            json.dump(global_synonyms, f, indent=4)
    
    # check if the caption contains the synonyms
    bias_captions = {}
    for bias_cluster in bias_captions_final:
        bias_captions[bias_cluster] = {}
        for bias in bias_captions_final[bias_cluster]:
            bias_captions[bias_cluster][bias] = {}
            current_words_to_check = words_to_check[bias_cluster][bias]
            for class_cluster in bias_captions_final[bias_cluster][bias]:
                bias_captions[bias_cluster][bias][class_cluster] = []
                for caption_id, question in bias_captions_final[bias_cluster][bias][class_cluster]:
                    caption = captions[caption_id][0].lower()
                    remove_caption = False
                    for word in current_words_to_check:
                        word = word.lower()
                        if word in caption.split():
                            remove_caption = True
                            break
                    if not remove_caption:
                        bias_captions[bias_cluster][bias][class_cluster].append(
                            (
                                caption_id, 
                                question
                            )
                        )

    return bias_captions

def filter_caption_real(
    captions,
    bias_classes_final, 
    bias_captions_final,
):
    return bias_captions_final

def merge_in_one_cluster(
    bias_classes_final,
    bias_captions_final
):
    bias_classes_merged = deepcopy(bias_classes_final)
    bias_captions_merged = deepcopy(bias_captions_final)
    for bias_cluster in bias_classes_final:
        for bias in bias_classes_final[bias_cluster]:
            # get max count class
            max_count_class, max_counts = get_max_count(bias_classes_final[bias_cluster][bias])
            # merge others and remove all other clusters
            for class_cluster in bias_classes_final[bias_cluster][bias]:
                if class_cluster != max_count_class:
                    bias_classes_merged[bias_cluster][bias][max_count_class]['counts'] += bias_classes_final[bias_cluster][bias][class_cluster]['counts']
                    bias_captions_merged[bias_cluster][bias][max_count_class] += bias_captions_final[bias_cluster][bias][class_cluster]
                    del bias_classes_merged[bias_cluster][bias][class_cluster]
                    del bias_captions_merged[bias_cluster][bias][class_cluster]
    return bias_classes_merged, bias_captions_merged

'''
    This function :
    - clusters the biases based on the refer_to
    - clusters the biases based on their name
    - clusters the biases based on their classes
        - this sub cluster is made by joining the classes with the JOIN_TOKEN
'''
def clustering(
    LLM_output,
    valid_bias_fn,
    JOIN_TOKEN
):
    # save caption and image id
    # captions[caption_id] = (caption, image_id)
    captions = {}

    # save image ids for each caption
    # image_ids[image_id] = [caption_id_1, caption_id_2, ...]
    image_ids = defaultdict(list)

    # group biases by group (refer_to)
    # bias_captions['person']['gender']['cluster_1'] = [
    #   (
    #       caption_id,
    #       question,
    #   )
    # ]
    bias_captions = defaultdict(nested_dict_list)

    # bias_classes['person']['gender']['cluster_1'] = {
    #   'classes': [class_1, class_2, ...],
    #   'counts': counts,
    # }
    bias_classes = defaultdict(nested_dict)

    # classes - cluster mapper
    # class_clustes['person']['gender']['string of classes'] = 'cluster_1'
    class_clusters = defaultdict(nested_dict_str)
    # cluster - classes string mapper (opposite of above)
    # class_clustes_string['person']['gender']['cluster_1'] = 'string of classes'
    class_clusters_string = defaultdict(nested_dict_str)
    # for each caption
    for caption in tqdm(LLM_output):
        # check if caption is valid
        if valid_caption(caption):
            if ',' in str(caption['image_id']):
                caption['image_id'] = caption['image_id'].replace(',', '')
            # save caption and image id
            captions[caption['caption_id']] = (caption['caption'], caption['image_id'])
            image_ids[caption['image_id']].append(caption['caption_id'])
            # for each bias
            for bias in caption['proposed_biases']['bias']:
                # check if bias is valid
                if valid_bias_fn(bias):
                    bias['name'] = bias['name'].lower()
                    bias_cluster = bias['refer_to'].lower()
                    classes = [str(c).lower().strip() for c in bias['classes']]
                    classes.sort()
                    classes_string = JOIN_TOKEN.join(classes)
                    class_cluster = class_clusters[bias_cluster][bias['name']][classes_string]
                    # if class cluster does not exist
                    if class_cluster == '':
                        # create new class cluster
                        class_cluster = 'cluster_' + str(len(list(class_clusters[bias_cluster][bias['name']].keys())))
                        class_clusters[bias_cluster][bias['name']][classes_string] = class_cluster
                        bias_classes[bias_cluster][bias['name']][class_cluster]['classes'] = classes
                        bias_classes[bias_cluster][bias['name']][class_cluster]['counts'] = 0
                        class_clusters_string[bias_cluster][bias['name']][class_cluster] = classes_string

                    # save caption and question
                    bias_captions[bias_cluster][bias['name']][class_cluster].append(
                        (
                            caption['caption_id'],
                            bias['question'],
                        )
                    )
                    bias_classes[bias_cluster][bias['name']][class_cluster]['counts'] += 1

    return captions, image_ids, bias_captions, bias_classes, class_clusters, class_clusters_string

'''
    This function :
    - filters the biases based on a threshold
'''
def filter_classes(
    bias_classes,
    bias_captions,
    threshold=0.5,
    hard_threshold=10
):
    bias_classes_filtered = defaultdict(nested_dict)
    bias_captions_filtered = defaultdict(nested_dict_list)
    to_add_captions = nested_dict_list()
    to_add_counts = {}
    for bias_cluster in bias_classes:
        to_add_counts[bias_cluster] = {}
        for bias in bias_classes[bias_cluster]:
            to_add_counts[bias_cluster][bias] = {'counts': 0}
            max_count_class, max_counts = get_max_count(bias_classes[bias_cluster][bias])
            for class_cluster in bias_classes[bias_cluster][bias]:
                # filter classes based on threshold
                if bias_classes[bias_cluster][bias][class_cluster]['counts'] / max_counts >= threshold and \
                    bias_classes[bias_cluster][bias][class_cluster]['counts'] >= hard_threshold:
                    bias_classes_filtered[bias_cluster][bias][class_cluster] = deepcopy(bias_classes[bias_cluster][bias][class_cluster])
                    bias_captions_filtered[bias_cluster][bias][class_cluster] = deepcopy(bias_captions[bias_cluster][bias][class_cluster])
                else:
                    # save data to be added to the first cluster
                    to_add_captions[bias_cluster][bias] += bias_captions[bias_cluster][bias][class_cluster]
                    to_add_counts[bias_cluster][bias]['counts'] += bias_classes[bias_cluster][bias][class_cluster]['counts']
    
    return bias_classes_filtered, bias_captions_filtered, to_add_captions, to_add_counts

'''
    This function merges common biases
    a common bias is defined with a class cluster overlap above a threshold
'''
def merge_class_clusters(
    bias_classes_filtered,
    bias_captions,
    class_clusters,
    class_clusters_string,
    merge_threshold=0.75,
    JOIN_TOKEN='<>',
):
    # merge common biases
    # initialize variables
    bias_classes_merged = deepcopy(bias_classes_filtered)
    bias_captions_merged = deepcopy(bias_captions)
    class_clusters_merged = deepcopy(class_clusters)
    class_clusters_string_merged = deepcopy(class_clusters_string)
    for bias_cluster in bias_classes_filtered:
        for bias in bias_classes_filtered[bias_cluster]:
            # list of clusters merged
            clusters_merged = []
            for idx, class_cluster in enumerate(bias_classes_filtered[bias_cluster][bias]):
                # if cluster was not already merged check it
                if class_cluster not in clusters_merged:
                    # get main class string
                    main_classes_string = class_clusters_string[bias_cluster][bias][class_cluster]
                    # get remaining clusters
                    remaining_clusters = list(bias_classes_filtered[bias_cluster][bias].keys())[idx+1:]
                    for remaining_cluster in remaining_clusters:
                        # if remaining cluster was not already merged check it
                        if remaining_cluster not in clusters_merged:
                            # retrieve current class string
                            current_class_string = class_clusters_string[bias_cluster][bias][remaining_cluster]
                            # compute overlap
                            overlap = compute_overlap(main_classes_string.split(JOIN_TOKEN), current_class_string.split(JOIN_TOKEN))
                            # if overlap is above threshold then merge clusters
                            if overlap >= merge_threshold:
                                # update counts
                                bias_classes_merged[bias_cluster][bias][class_cluster]['counts'] += bias_classes_filtered[bias_cluster][bias][remaining_cluster]['counts']
                                # merge classes
                                classes_set = set(bias_classes_filtered[bias_cluster][bias][class_cluster]['classes'])
                                classes_set.update(bias_classes_filtered[bias_cluster][bias][remaining_cluster]['classes'])
                                bias_classes_merged[bias_cluster][bias][class_cluster]['classes'] = list(classes_set)
                                # merge captions
                                bias_captions_merged[bias_cluster][bias][class_cluster].extend(bias_captions[bias_cluster][bias][remaining_cluster])
                                # add merged cluster to the list of merged clusters
                                clusters_merged.append(remaining_cluster)
                                # remove merged cluster
                                del bias_classes_merged[bias_cluster][bias][remaining_cluster]
                                del bias_captions_merged[bias_cluster][bias][remaining_cluster]
                                # remove old class cluster
                                del class_clusters_merged[bias_cluster][bias][current_class_string]
                                del class_clusters_merged[bias_cluster][bias][main_classes_string]
                                del class_clusters_string_merged[bias_cluster][bias][class_cluster]
                                del class_clusters_string_merged[bias_cluster][bias][remaining_cluster]
                                # add new class clusters
                                new_classes_string = JOIN_TOKEN.join(bias_classes_merged[bias_cluster][bias][class_cluster]['classes'])
                                class_clusters_string_merged[bias_cluster][bias][class_cluster] = new_classes_string
                                class_clusters_merged[bias_cluster][bias][new_classes_string] = class_cluster
                                # update main class string
                                main_classes_string = new_classes_string

    return bias_classes_merged, bias_captions_merged, class_clusters_merged, class_clusters_string_merged

def remove_duplicated_biases(
    bias_classes_merged,
    bias_captions_merged,
):
    bias_captions_final = deepcopy(bias_captions_merged)
    bias_classes_final = deepcopy(bias_classes_merged)

    for bias_cluster in bias_captions_merged:
        for idx, bias_name in enumerate(bias_captions_merged[bias_cluster]):
            # check if the current bias name has not been removed previously
            if bias_name in list(bias_captions_final[bias_cluster].keys()):
                # split bias name
                attribute_1 = bias_name.replace(bias_cluster, '').strip()
                # for each other bias name
                for i in range(idx+1, len(list(bias_captions_merged[bias_cluster].keys()))):
                    remaining_bias_name = list(bias_captions_merged[bias_cluster].keys())[i]
                    # check if the remaining bias name has not been removed previously
                    if remaining_bias_name in list(bias_captions_final[bias_cluster].keys()):
                        # split remaining bias name
                        attribute_2 = remaining_bias_name.replace(bias_cluster, '').strip()
                        # compute similarity
                        attribute_1_set = set(attribute_1.split())
                        attribute_2_set = set(attribute_2.split())
                        common = attribute_1_set & attribute_2_set
                        # if one word in common
                        if len(attribute_2_set) == 1 and len(common) > 0:
                            # remove the bias with the same classes
                            del bias_captions_final[bias_cluster][remaining_bias_name]
                            del bias_classes_final[bias_cluster][remaining_bias_name]

    return bias_classes_final, bias_captions_final   

'''
    This function :
    - post processes the LLM output file
    - saves the bias classes
'''
def post_processing(
    LLM_output_file,
    threshold,
    hard_threshold,
    merge_threshold,
    valid_bias_fn,
    filter_caption_fn,
    all_images,
):
    '''
    captions[caption_id] = (caption, image_id)
    
    image_ids[image_id] = [caption_id_1, caption_id_2, ...]
    group biases by group (refer_to)
    bias_captions['person']['gender']['cluster_1'] = [
      (
          caption_id,
          question,
      )
    ]

    bias_classes['person']['gender']['cluster_1'] = {
      'classes': [class_1, class_2, ...],
      'counts': counts,
    }

    classes - cluster mapper
    class_clustes['person']['gender']['string of classes'] = 'cluster_1'
    cluster - classes string mapper (opposite of above)
    class_clustes_string['person']['gender']['cluster_1'] = 'string of classes'
    '''
    
    # read file
    with open(LLM_output_file, 'r') as f:
        LLM_output = json.load(f)['bias_proposal']
    
    # join token for class clusters
    JOIN_TOKEN = '<>'
    # cluster biases
    captions, image_ids, bias_captions, bias_classes, class_clusters, class_clusters_string = clustering(
        LLM_output,
        valid_bias_fn,
        JOIN_TOKEN,
    )

    # filter biases with classes cluster below threshold
    bias_classes_filtered, bias_captions_filtered, to_add_captions, to_add_counts = filter_classes(
        bias_classes,
        bias_captions,
        threshold=threshold,
        hard_threshold=hard_threshold,
    )

    # merge common classes clusters
    bias_classes_merged, bias_captions_merged, class_clusters_merged, class_clusters_string_merged = merge_class_clusters(
        bias_classes_filtered,
        bias_captions_filtered,
        class_clusters,
        class_clusters_string,
        merge_threshold=merge_threshold,
        JOIN_TOKEN=JOIN_TOKEN,
    )

    # remove duplicate biases
    bias_classes_final, bias_captions_final = remove_duplicated_biases(
        bias_classes_merged,
        bias_captions_merged,
    )

    # Add filtered captions to the first cluster of each bias
    if all_images:
        for bias_cluster in to_add_captions:
            for bias in to_add_captions[bias_cluster]:
                if len(list(bias_captions_final[bias_cluster][bias].keys())) > 0:
                    bias_captions_final[bias_cluster][bias][
                        list(bias_captions_final[bias_cluster][bias].keys())[0]
                    ] += to_add_captions[bias_cluster][bias]
                    bias_classes_final[bias_cluster][bias][
                        list(bias_captions_final[bias_cluster][bias].keys())[0]
                    ]['counts'] += to_add_counts[bias_cluster][bias]['counts']

    # filter captions with synonyms
    bias_captions_final = filter_caption_fn(
        captions,
        bias_classes_final,
        bias_captions_final
    )

    # merge in one single cluster (to the one with the most counts)
    bias_classes_final, bias_captions_final = merge_in_one_cluster(
        bias_classes_final,
        bias_captions_final
    )

    return captions, image_ids, bias_classes_final, bias_captions_final, class_clusters_merged, class_clusters_string_merged

'''
    This function :
    - takes the first caption for each real image
'''
def get_first_caption(
    captions_id,
    captions,
    max_prompts,
):
    image_ids_done = []
    captions_id_final = []
    for caption_id, question in captions_id:
        image_id = captions[caption_id][1]
        if image_id not in image_ids_done:
            image_ids_done.append(image_id)
            captions_id_final.append((caption_id, question))
        if len(captions_id_final) == max_prompts:
            break
    return captions_id_final

'''
    This function :
    - return the image name with the correct format (12 digits)
'''
def get_image_name_coco(image_id):
    return str(image_id).rjust(12, '0')+'.jpg'

'''
    This function :
    - return the image name with the correct format
'''
def get_image_name_open_images(image_id):
    return str(image_id)+'.jpg'

'''
    This function :
    - return the image name with the correct format
'''
def get_image_name_flickr(image_id):
    return image_id