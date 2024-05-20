from torch.utils.data import Dataset
import os
import json
import csv
from pycocotools.coco import COCO
import utils.utils as utils
from collections import defaultdict
from copy import deepcopy
from PIL import Image
import torchvision.transforms as T
import fiftyone
# from openimages.download import download_images as open_images_download

#####################################################################################
#######                                                                       #######
#######                        BIAS PROPOSAL DATASETS                         #######
#######                                                                       #######
#####################################################################################
class Coco(Dataset):
    def __init__(
        self,
        opt
    ):
        path = opt['dataset_setting']['path']
        mode = opt['dataset_setting']['mode']
        categories = opt['dataset_setting']['categories']
        self.n_prompts_per_image = opt['dataset_setting']['n_prompts_per_image']

        path_annotations = os.path.join(path, 'annotations')
        path_images = os.path.join(path, 'images')
        if mode == '_train':
            path_instances = os.path.join(path_annotations, 'instances_train2017.json')
            path_annotations = os.path.join(path_annotations, 'captions_train2017.json')
            path_images = os.path.join(path_images, 'train2017')
        elif mode == '_val':
            path_instances = os.path.join(path_annotations, 'instances_val2017.json')
            path_annotations = os.path.join(path_annotations, 'captions_val2017.json')
            path_images = os.path.join(path_images, 'val2017')
        elif mode == '_test':
            path_images = os.path.join(path_images, 'test2017')
        
        self.path_annotations = path_annotations
        self.path_instances = path_instances
        self.path_images = path_images
        # filter images based on the category
        self.filtered_image_ids = self.filter_image_ids(categories)
        # load images
        self.captions, self.image_ids, self.caption_ids = self._get_annotations()
        self.captions = self.captions[:100]
        self.image_ids = self.image_ids[:100]
        self.caption_ids = self.caption_ids[:100]

    def _get_annotations(self):
        with open(self.path_annotations, 'r') as f:
            data = json.load(f)
        captions = []
        real_image_counts = defaultdict(int)
        caption_ids = []
        image_ids = []
        for annotation in data['annotations']:
            image_id = int(annotation['image_id'])
            if image_id in self.filtered_image_ids:
                if real_image_counts[image_id] < self.n_prompts_per_image:
                    captions.append(annotation['caption'])
                    caption_ids.append(int(annotation['id']))
                    image_ids.append(image_id)
                    real_image_counts[image_id] += 1

        return captions, image_ids, caption_ids

    def filter_image_ids(self, categories):
        # get category ids
        coco = COCO(self.path_instances)
        category_ids = coco.getCatIds(catNms=categories)

        # retrieve single person images only
        with open(self.path_instances, 'r') as f:
            data = json.load(f)
        # count number of annotations per image
        image_ids_counts = {}
        for annotation in data['annotations']:
            img_id = int(annotation['image_id'])
            category_id = int(annotation['category_id'])
            # filter out non person images
            # there may be multiple annotations for each image -> counts
            if category_id in category_ids:
                if img_id not in image_ids_counts:
                    image_ids_counts[img_id] = 0
                image_ids_counts[img_id] += 1
        # retain the images with one annotation only (single person)
        filtered_image_ids = [image_id for image_id in image_ids_counts if image_ids_counts[image_id] == 1]
        return filtered_image_ids
    
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return self.captions[index], self.image_ids[index], self.caption_ids[index]

class Flickr_30k(Dataset):
    def __init__(
        self,
        opt
    ) -> None:
        path = opt['dataset_setting']['path']
        self.n_prompts_per_image = opt['dataset_setting']['n_prompts_per_image']
        self.path_annotations = os.path.join(path, 'captions.txt')
        self.path_images = os.path.join(path, 'Images')
        self.captions, self.img_ids = self._get_annotations()
        self.images = os.listdir(self.path_images)
    
    def get_images(self):
        return self.images
    
    def _get_annotations(self):
        with open (self.path_annotations, "r") as myfile:
            data = myfile.read().splitlines()[1:] 
        captions = []
        img_ids = []
        real_image_counts = defaultdict(int)
        for line in data:
            if '"' in line:
                caption = line.split('"')[1].strip()
                img_id = line.split('"')[0].strip().replace(",", '')
            else:
                caption = line.split(",")[1].replace('"', '').strip()
                img_id = line.split(", ")[0]

            # get first <n_prompts_per_image> captions for each real image
            if real_image_counts[img_id] < self.n_prompts_per_image:
                img_id = img_id.replace(',', '')
                captions.append(caption)
                img_ids.append(img_id)
                real_image_counts[img_id] += 1

        return captions, img_ids

    def __len__(self):
        return len(self.captions)
    
    def get_captions(self):
        return self.captions, self.img_ids

    def __getitem__(self, index):
        return self.captions[index], self.img_ids[index]

'''
    This dataset is used to caption the images from a generic image dataset (e.g., generated imgaes from stylegan3)
'''
class Image_dataset(Dataset):
    def __init__(self, data, dataset_path):
        self.data = data
        self.dataset_path = dataset_path
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_path, self.data[idx], '0.png')
        image = Image.open(image_path)
        return image, self.data[idx]

class Captioned_dataset(Dataset):
    def __init__(
        self,
        opt
    ) -> None:
        path = opt['dataset_setting']['path']
        self.data = []
        with open(os.path.join(path, 'captions.json'), 'r') as f:
            captions = json.load(f)
        for image_id in captions:
            self.data.append(
                (
                    captions[image_id],
                    image_id
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]
    
class WinoBias:
    def __init__(
        self,
        opt
    ):
        path = opt['dataset_setting']['path']
        with open(os.path.join(path, 'professions.txt'), 'r') as f:
            self.professions = f.readlines()
        
        self.professions = [profession.strip() for profession in self.professions]
        self.captions = []
        j = 0
        for i in range(len(self.professions)):
            self.captions.append(
                (
                    f'A person working as {self.professions[i]}. JSON.',
                    str(j)
                )
            )
            j += 1
            self.captions.append(
                (
                    f'A person who is a {self.professions[i]}. JSON.',
                    str(j)
                )
            )
            j += 1
            self.captions.append(
                (
                    f'A {self.professions[i]}. JSON.',
                    str(j)
                )
            )
            j += 1
            self.captions.append(
                (
                    f'A human working as {self.professions[i]}. JSON.',
                    str(j)
                )
            )
            j += 1
    
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return self.captions[index][0], self.captions[index][1], self.captions[index][1]  

#####################################################################################
#######                                                                       #######
#######                     IMAGE GENERATION DATASETS                         #######
#######                                                                       #######
#####################################################################################
class Proposed_biases(Dataset):
    def __init__(
        self,
        dataset_path,
        max_prompts,
        filter_threshold,
        hard_threshold,
        merge_threshold,
        valid_bias_fn,
        filter_caption_fn,
        all_images
    ):
        super(Proposed_biases).__init__()
        self.max_prompts = max_prompts

        print("Loading and filtering proposed biases...")
        # post process LLM output
        captions, image_ids, bias_classes_final, bias_captions_final, class_clusters_merged, class_clusters_string_merged = utils.post_processing(
            dataset_path,
            threshold=filter_threshold,
            hard_threshold=hard_threshold,
            merge_threshold=merge_threshold,
            valid_bias_fn=valid_bias_fn,
            filter_caption_fn=filter_caption_fn,
            all_images=all_images
        )
        print("Done!")
        # prompts to use for image generation 
        # we take one <max_prompts> 
        self.prompts = set()
        # for each bias group
        for bias_group_name in bias_captions_final:
            # for each bias
            for bias_name in bias_captions_final[bias_group_name]:
                # for each class cluster
                for class_cluster in bias_captions_final[bias_group_name][bias_name]:
                    # get first caption captions
                    captions_ids = utils.get_first_caption(
                        captions_id = bias_captions_final[bias_group_name][bias_name][class_cluster],
                        captions = captions,
                        max_prompts = max_prompts
                    )
                    # for each caption
                    for caption_id, question in captions_ids:
                        # get caption and image id
                        caption, image_id = captions[caption_id]
                        # add prompt
                        self.prompts.add((caption, caption_id))

        self.prompts = list(self.prompts)
        self.prompts = sorted(self.prompts, key=lambda x: x[1])
        self.bias_captions_final = bias_captions_final

        self.bias_classes_final = bias_classes_final
        self.captions = captions

    def __len__(self):
        return len(self.prompts)

    def get_bias_captions_id(self):
        return self.bias_captions_id

    def get_bias_classes(self):
        return self.bias_classes_final

    def get_biases(self):
        return self.bias_captions_final, self.bias_classes_final, self.captions

    def get_data(self):
        return self.prompts

    def __getitem__(self, index):
        caption, caption_id = self.prompts[index]
        return caption, caption_id

#####################################################################################
#######                                                                       #######
#######                             VQA DATASETS                              #######
#######                                                                       #######
#####################################################################################
class VQA_dataset(Dataset):
    def __init__(
        self,
        dataset_setting,
        mode,
        max_prompts,
        filter_threshold,
        hard_threshold,
        merge_threshold,
        valid_bias_fn,
        filter_caption_fn
    ):
        super().__init__()
        images_paths = dataset_setting['images_path']
        proposed_biases_path = dataset_setting['proposed_biases_path']
        self.images_paths = images_paths
        self.mode = mode
        # get predicted biases
        bias_dataset = Proposed_biases(
            proposed_biases_path,
            max_prompts,
            filter_threshold,
            hard_threshold,
            merge_threshold,
            valid_bias_fn,
            filter_caption_fn,
            dataset_setting['all_images']
        )
        # get biases and captions
        bias_captions_final, bias_classes_final, captions = bias_dataset.get_biases()

        # define dict of biases
        # biases = {
        #     'caption_id': [
        #         (
        #            bias_cluster,
        #            bias_name,
        #            classes_cluster,
        #            question,
        #            [classes]
        #         ),
        #        ...    
        #     ]
        # }
        biases = defaultdict(list)
        # for each bias cluster
        for bias_cluster in bias_captions_final:
            # for each bias
            for bias_name in bias_captions_final[bias_cluster]:
                # for each class cluster
                for class_cluster in bias_captions_final[bias_cluster][bias_name]:
                    # get first caption for each real image
                    cpts = utils.get_first_caption(
                        captions_id = bias_captions_final[bias_cluster][bias_name][class_cluster],
                        captions = captions,
                        max_prompts = max_prompts
                    )
                    # for each caption
                    for cpt_id, question in cpts:
                        # add bias information to the dict of caption ids
                        biases[cpt_id].append(
                            (
                                bias_cluster,
                                bias_name,
                                class_cluster,
                                question,
                                bias_classes_final[bias_cluster][bias_name][class_cluster]['classes']
                            )
                        )
        
        self.biases = biases

        if self.mode == 'generated':
            # define data
            # data = [
            #     (
            #         caption_id,
            #         caption,
            #         image_path,
            #         [
            #             (
            #                 bias_cluster,
            #                 bias_name,
            #                 class_cluster,
            #                 question,
            #                 [classes]
            #             ),
            #             ...
            #         ]
            #     ),
            #     ...
            # ]
            self.data = []
            # for each caption id
            for caption_id in biases:
                # get images path
                image_path = os.path.join(images_paths, str(caption_id))
                # get list of images
                images = os.listdir(image_path)
                # for each image
                for image_name in images:
                    # save bias information regarding the image
                    self.data.append(
                        (
                            caption_id, 
                            captions[caption_id][0],
                            captions[caption_id][1],
                            os.path.join(images_paths, str(caption_id), image_name), 
                            biases[caption_id]
                        )
                    )
        elif self.mode == 'original':
            # group biases by image id
            # it may happen that the same real image is associated with multiple captions coming from different biases
            image_ids = defaultdict(list)
            for caption_id in biases:
                image_id = captions[caption_id][1]
                image_ids[image_id] += biases[caption_id]
                
            # define data
            # data = [
            #     (
            #         image_id,
            #         image_path,
            #         [
            #             (
            #                 bias_cluster,
            #                 bias_name,
            #                 class_cluster,
            #                 question,
            #                 [classes]
            #             ),
            #             ...
            #         ]
            #     ),
            #     ...
            # ]
            self.data = []
            for image_id in image_ids:
                # get image path
                image_path = os.path.join(
                    images_paths, 
                    dataset_setting['get_image_name'](image_id)
                )
                # save bias information regarding the image
                self.data.append(
                    (
                        image_id, 
                        image_path, 
                        image_ids[image_id]
                    )
                )

        # classes
        self.bias_classes_final = bias_classes_final
        
    def __len__(self):
        return len(self.data)

    def get_bias_classes(self):
        return self.bias_classes_final

    def get_classes(self):
        return self.classes

    def get_data(self):
        return self.data
    
    def get_proposed_biases_per_caption(self):
        return self.biases

    def __getitem__(self, index):
        if self.mode == 'generated':
            # get image info
            caption_id, caption, image_id, image_path, proposed_biases = self.data[index]
            return caption_id, caption, image_id, image_path, proposed_biases
        elif self.mode == 'original':
            # get image info
            image_id, image_path, proposed_biases = self.data[index]
            return None, None, image_id, image_path, proposed_biases

class VQA_specific_biases(Dataset):
    def __init__(
        self,
        dataset_setting,
        mode,
        max_prompts,
        filter_threshold,
        hard_threshold,
        merge_threshold,
        valid_bias_fn,
        biases
    ):
        super().__init__()
        images_paths = dataset_setting['images_path']
        proposed_biases_path = dataset_setting['proposed_biases_path']
        self.images_paths = images_paths
        self.mode = mode
        # get predicted biases
        bias_dataset = Proposed_biases(
            proposed_biases_path,
            max_prompts,
            filter_threshold,
            hard_threshold,
            merge_threshold,
            valid_bias_fn,
            dataset_setting['filter_caption_fn'],
            dataset_setting['all_images']
        )
        # get biases and captions
        bias_captions_final, bias_classes_final, captions = bias_dataset.get_biases()

        specific_bias = {}
        # for each bias cluster
        for bias_cluster in bias_captions_final:
            # for each bias
            for bias_name in bias_captions_final[bias_cluster]:
                if bias_name in biases or len(biases)==0:
                    class_clusters = list(bias_captions_final[bias_cluster][bias_name].keys())
                    # sort by number of captions
                    class_clusters = sorted(class_clusters, key=lambda x: len(bias_captions_final[bias_cluster][bias_name][x]), reverse=True)
                    # get first cluster 
                    class_cluster = class_clusters[0]
                    
                    cpts = utils.get_first_caption(
                        captions_id = bias_captions_final[bias_cluster][bias_name][class_cluster],
                        captions = captions,
                        max_prompts = max_prompts
                    )
                    specific_bias[bias_name] = {
                        'classes': bias_classes_final[bias_cluster][bias_name][class_cluster]['classes']
                    }
                    # for each caption
                    images = []
                    for cpt_id, question in cpts:
                        image = os.listdir(os.path.join(images_paths, str(cpt_id)))
                        for img in image:
                            images.append(
                                os.path.join(images_paths, str(cpt_id), img)
                            )
                    specific_bias[bias_name]['images'] = images
        self.specific_bias = specific_bias
    
    def __len__(self):
        return len(self.specific_bias)

    def get_specific_bias(self):
        return self.specific_bias

#####################################################################################
#######                                                                       #######
#######                             VQA EVALUATION                            #######
#######                                                                       #######
#####################################################################################

class VQA_evaluation(Dataset):
    def __init__(
        self,
        mode,
        dataset_setting,
        bias_name,
        bias_cluster,
        max_prompts,
        filter_threshold,
        hard_threshold,
        merge_threshold
    ):
        super().__init__()
        proposed_biases_path = dataset_setting['proposed_biases_path']
        images_path = dataset_setting['images_path']
        # get predicted biases
        ds = Proposed_biases(
            proposed_biases_path,
            max_prompts,
            filter_threshold,
            hard_threshold,
            merge_threshold
        )
        # get biases and captions
        bias_captions_final, bias_classes_final, captions = ds.get_biases()

        # get first captions
        cpts = utils.get_first_caption(
            captions_id = bias_captions_final[bias_cluster][bias_name][list(bias_captions_final[bias_cluster][bias_name].keys())[0]],
            captions = captions,
            max_prompts = max_prompts
        )

        # get images and questions
        self.data = []
        for caption_id, question in cpts:
            if mode == 'generated':
                images = os.listdir(os.path.join(images_path, str(caption_id)))
                for image_name in images:
                    image = os.path.join(images_path, str(caption_id), image_name)
                    self.data.append(
                        (
                            image,
                            question
                        )
                    )
            elif mode == 'original':
                image_id = captions[caption_id][1]
                image = os.path.join(
                    images_path, 
                    dataset_setting['get_image_name'](image_id)
                )
                self.data.append(
                    (
                        image,
                        question
                    )
                )
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, question = self.data[idx]
        return image, question

class FairFace_Reference(Dataset):
    def __init__(
        self,
        dataset_setting,
        mode,
        max_prompts,
        filter_threshold,
        hard_threshold,
        merge_threshold,
        valid_bias_fn,
        bias_cluster,
        bias_name
    ):
        super().__init__()
        images_paths = dataset_setting['images_path']
        proposed_biases_path = dataset_setting['proposed_biases_path']
        self.mode = mode
        # get predicted biases
        bias_dataset = Proposed_biases(
            proposed_biases_path,
            max_prompts,
            filter_threshold,
            hard_threshold,
            merge_threshold,
            valid_bias_fn,
            dataset_setting['all_images']
        )
        # get biases and captions
        bias_captions_final, bias_classes_final, captions = bias_dataset.get_biases()

        bias_captions_final = bias_captions_final[bias_cluster][bias_name][
            list(bias_captions_final[bias_cluster][bias_name].keys())[0]
        ]

        bias_classes_final = bias_classes_final[bias_cluster][bias_name][
            list(bias_classes_final[bias_cluster][bias_name].keys())[0]
        ]

        cpts = utils.get_first_caption(
            captions_id = bias_captions_final,
            captions = captions,
            max_prompts = max_prompts
        )

        # get images and questions
        self.data = []
        for caption_id, question in cpts:
            if mode == 'generated':
                images = os.listdir(os.path.join(images_paths, str(caption_id)))
                for image_name in images:
                    image = os.path.join(images_paths, str(caption_id), image_name)
                    self.data.append(
                        (
                            image,
                            question
                        )
                    )
            elif mode == 'original':
                image_id = captions[caption_id][1]
                image = os.path.join(
                    images_paths, 
                    dataset_setting['get_image_name'](image_id)
                )
                self.data.append(
                    (
                        image,
                        question
                    )
                )
        
        self.transform = T.Compose([
            T.Resize((500, 300)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, _ = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        return image, image_path