# Step 1: propose biases leveraging a specific LLM (Llama2)
CUDA_VISIBLE_DEVICES=0 python bias_proposals.py --workers 6 --dataset 'coco' 

# Step 2: generate images using a specific target generative model
# CUDA_VISIBLE_DEVICES=0 python generate_images.py --dataset coco --generator sd-xl

# Step 3: run VQA on the generated images
# CUDA_VISIBLE_DEVICES=0 python run_VQA.py --vqa_model llava-1.5-13b --workers 4 --dataset 'coco' --mode 'generated' --generator sd-xl
# CUDA_VISIBLE_DEVICES=0 python run_VQA.py --vqa_model llava-1.5-13b --workers 4 --dataset 'coco' --mode 'original'

# Generate images using stylegan3 - unconditional generation
# CUDA_VISIBLE_DEVICES=0 python StyleGan3_generation.py --n_images 100 --generator stylegan3-ffhq
# CUDA_VISIBLE_DEVICES=0 python captioning.py --generator stylegan3-ffhq 

# Step 4: plot results
# python make_plots.py --generator sd-xl --dataset coco --mode generated
