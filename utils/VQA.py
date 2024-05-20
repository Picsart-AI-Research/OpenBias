import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from utils.bert import SBERTModel
import modelscope
from transformers import AutoProcessor, AutoModelForCausalLM, BlipForQuestionAnswering, ViltForQuestionAnswering
from lavis.models import load_model_and_preprocess
from modelscope.outputs import OutputKeys
from promptcap import PromptCap_VQA
import clip, sys

# llava model imports
from utils.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from utils.llava.conversation import conv_templates, SeparatorStyle
from utils.llava.model.builder import load_pretrained_model
from utils.llava.utils import disable_torch_init
from utils.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import open_clip
from torchvision import transforms
import torch.nn.functional as F

#####################################################################################
#######                                                                       #######
#######                               VQA                                     #######
#######                                                                       #######
#####################################################################################
class VQA():
    def __init__(
        self,
        device,
        opt
    ):
        opt['logger'].info(f'Loading VQA model: {opt["vqa_model"]}')
        class_name, self.model_path = opt['vqa_model']
        self.model = eval(class_name)(device=device, ckpt=self.model_path)
        opt['logger'].info(f'VQA model loaded')
        opt['logger'].info(f'Loading SBERT model')
        self.sbert_model = SBERTModel("sentence-transformers/all-mpnet-base-v2")
        opt['logger'].info(f'SBERT model loaded')
    
    @torch.no_grad()
    def get_caption(self, image, question):
        # Get VQA model's answer
        caption = self.model.vqa(
            image=image, 
            question=question
        )
        return caption

    @torch.no_grad()
    def process_image(self, image):
        image = image.convert('RGB')
        return self.model.process_image(image)

    @torch.no_grad()
    def get_free_form_answer(self, image, question):
        free_form_answer = self.model.vqa(
            image=image, 
            question=question,
        )
        return free_form_answer
    
    @torch.no_grad()
    def get_answer(self, image, question, choices):
        # Get VQA model's answer
        free_form_answer = self.model.vqa(
            image=image, 
            question=question, 
            choices=choices
        )
        
        # Limit the answer to the choices
        multiple_choice_answer = free_form_answer
        if free_form_answer not in choices:
            multiple_choice_answer = self.sbert_model.multiple_choice(free_form_answer, choices)
        return {"free_form_answer": free_form_answer, "multiple_choice_answer": multiple_choice_answer}

    def multi_question(self, image, questions):
        answers = []
        for question, choices in questions:
            answers.append(self.get_answer(image, question, choices)['multiple_choice_answer'])
        return answers

class MPLUG():
    def __init__(self, device, ckpt):
        self.vqa_model = pipeline(Tasks.visual_question_answering, model=ckpt, device=device)

    def vqa(self, **kwargs):
        input_vqa = {'image': kwargs['image'], 'question': kwargs['question']}
        result = self.vqa_model(input_vqa)
        return result['text']

class GIT:
    def __init__(self, device, ckpt="microsoft/git-large-vqav2"):
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(ckpt)
        self.device = device
        self.model.to(self.device)
        
    def vqa(self, **kwargs):
        image = kwargs['image'].convert('RGB')
        question = kwargs['question']
        # prepare image
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        # prepare question
        input_ids = self.processor(text=question, add_special_tokens=False).input_ids
        input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        input_ids = input_ids.to(self.device)
    
        input_len = input_ids.shape[-1]
    
        generated_ids = self.model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        generated_ids = generated_ids[..., input_len:]
    
        generated_answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
   
        return generated_answer[0]

class BLIP2:
    def __init__(self, device, ckpt='pretrain_flant5xl'):
        self.device = device
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type=ckpt, is_eval=True, device=self.device)
        
    def vqa(self, **kwargs):
        image = kwargs['image'].convert('RGB')
        question = kwargs['question']
        choices = kwargs['choices']
        # prepare image
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        
        if len(choices) == 0:
            answer = self.model.generate({"image": image, "prompt": f"Question: {question} Answer:"})
        else:
            answer = self.model.generate({"image": image, "prompt": f"Question: {question} Choices: {', '.join(choices)}. Answer:"})
        return answer[0]

class BLIP:
    def __init__(self, device, ckpt="Salesforce/blip-vqa-capfilt-large"):
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = BlipForQuestionAnswering.from_pretrained(ckpt)
        self.device = device
        self.model.to(self.device)

    def vqa(self, **kwargs):
        image = kwargs['image'].convert('RGB')
        question = kwargs['question']
        # prepare image + question
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_length=50)
        generated_answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
       
        return generated_answer[0]

class VILT:
    def __init__(self, device, ckpt="dandelin/vilt-b32-finetuned-vqa"):
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = ViltForQuestionAnswering.from_pretrained(ckpt)
        self.device = device
        self.model.to(self.device)

    def vqa(self, **kwargs):
        image = kwargs['image'].convert('RGB')
        question = kwargs['question']

        # prepare image + question
        encoding = self.processor(images=image, text=question, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)

        predicted_class_idx = outputs.logits.argmax(-1).item()
    
        return self.model.config.id2label[predicted_class_idx]

class OFA:
    def __init__(self, device, ckpt='damo/ofa_visual-question-answering_pretrain_large_en'):
        from modelscope.outputs import OutputKeys
        from modelscope.preprocessors.multi_modal import OfaPreprocessor
        preprocessor = OfaPreprocessor(model_dir=ckpt)
        self.ofa_pipe = pipeline(
            Tasks.visual_question_answering,
            model=ckpt,
            preprocessor=preprocessor)
        
    def vqa(self, **kwargs):
        image = kwargs['image']
        question = kwargs['question']
        question = question.lower()
        input = {'image': image, 'text': question}
        result = self.ofa_pipe(input)
        return result[OutputKeys.TEXT][0]

class PromptCap:
    def __init__(self, device, ckpt='vqascore/promptcap-coco-vqa'):
        self.vqa_model = PromptCap_VQA(promptcap_model=ckpt, qa_model="allenai/unifiedqa-v2-t5-large-1363200")
        self.vqa_model.to(device)
        
    def vqa(self, **kwargs):
        return self.vqa_model.vqa(kwargs['question'], kwargs['image'])

# llava model
class Llava():
    def __init__(
        self,
        device,
        ckpt,
        model_base = None
    ):
        # Model
        disable_torch_init()
        self.model_name = get_model_name_from_path(ckpt)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(ckpt, model_base, self.model_name)
    
    def process_image(self, image):
        return self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    def vqa(self, **kwargs):
        if 'choices' in kwargs:
            qs = f'Question: {kwargs["question"]} Choices: {", ".join(kwargs["choices"])}. Answer:'
        else:
            qs = f'Question: {kwargs["question"]} Answer:'
            
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if 'llama-2' in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = kwargs['image']

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        return outputs

# clip
class Clip_model():
    def __init__(self, device, ckpt):
        print("-> Initializing CLIP...")
        self.device = device
        self.clip_model, self.clip_transform = clip.load(ckpt, device=device)
        self.clip_model.to(self.device)
        self.clip_model.float()
        self.clip_model.eval()
        print("---> CLIP Initialized")
    
    def compute_prob(self, image, text):
        image = self.clip_transform(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(text).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs

    def get_text_features(self, text):
        text_features = self.clip_model.encode_text(clip.tokenize(text).to(self.device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def vqa(self, **kwargs):
        image = kwargs['image']
        classes = kwargs['choices']
        probs = self.compute_prob(image, classes).tolist()[0]
        return classes[probs.index(max(probs))]

class Open_clip():
    def __init__(self, device, ckpt):
        print("-> Initializing CLIP...")
        self.device = device
        model_name = "ViT-B-32"
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=ckpt)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.clip_model.to(self.device)
        self.clip_model.float()
        self.clip_model.eval()
        print("---> CLIP Initialized")
        self.clip_img_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711)
            )
        ])
    
    def cosine_similarity(self, vector1, vector2):
        # Compute the cosine similarity between the two vectors
        cos_sim = torch.nn.functional.cosine_similarity(vector1, vector2, dim=0)
        return cos_sim.item()

    @torch.no_grad()
    def compute_text_embeddings(self, text):
        encoded_text = self.clip_model.encode_text(self.tokenizer(text).to(self.device))
        encoded_text /= torch.norm(encoded_text, dim=1, keepdim=True)
        return encoded_text
    
    def compute_text_cosine_similarity(self, text):
        cosine_similarities = []
        for prompt in text:
            encoded_text = self.compute_text_embeddings(prompt)
            cos_sim = []
            for idx in range(1,len(prompt)):
                cos_sim.append(self.cosine_similarity(encoded_text[0], encoded_text[idx]))
            cosine_similarities.append(cos_sim)
        return cosine_similarities

    @torch.no_grad()
    def compute_image_embedding(self, image):
        image_embedding = self.clip_model.encode_image(self.preprocess(image).unsqueeze(0).to(self.device))
        return F.normalize(image_embedding, p=2, dim=-1).to(self.device)

    def compute_prob(self, image, text):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(text).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs
    
    def vqa(self, **kwargs):
        image = kwargs['image']
        classes = kwargs['choices']
        probs = self.compute_prob(image, classes).tolist()[0]
        return classes[probs.index(max(probs))]