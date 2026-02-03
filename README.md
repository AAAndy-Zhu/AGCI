<h1 align="center"> Adaptive Global Context Injection (AGCI) </h1>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2509.21984-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.21984) <a href='https://huggingface.co/datasets/AAAndyZ/BaPA_Probe_Dataset'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Probe Dataset-green'></a> <a href='https://huggingface.co/datasets/AAAndyZ/BaPA_Similarity_Dataset'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Similarity Dataset-blue'></a>
</h5>

This repository contains the implementation of [*Beyond the Vision Encoder: Identifying and Mitigating Spatial Bias in  Large Vision-Language Models*](https://arxiv.org/abs/2509.21984). In this paper, we systematically investigates spatial robustness in Large Vision-Language Models (LVLMs) and propose **Adaptive Global Context Injection (AGCI)**, a simple yet effective method to mitigate spatial bias in LVLMs to improve cross-modal reasoning.


### Visualization of Information Flow 
To analyze the effect of AGCI on cross-modal interactions, we present a case study from TextVQA, with the **information flow** between image tokens and text tokens in LVLMs.

The baseline Qwen2.5-VL exhibits diffused and weak attention over the image, where the highest attention is incorrectly concentrated on the upper-left region of the image, which is irrelevant to the question, leading to an incorrect prediction. By contrast, after applying AGCI, attention becomes more concentrated on the regions containing the key textual cues, especially for the region containing the word **“British”** (the brightest patch), which serves as the critical OCR cue for answering the question.

<div align="center">
<img src="https://github.com/AAAndy-Zhu/AGCI/blob/main/case_study.png" width="700px" alt="Information Flow">
</div>

## 📊 Data
### Probe Dataset
We design a probe dataset by randomly sampling 10,000 image-caption pairs from LAION. Each image is placed in different spatial positions in a 3×3 grid combined with distractor images, enabling us to evaluate the robustness of LVLMs to spatial variations.

- **The dataset is publicly available on [🤗Hugging Face](https://huggingface.co/datasets/AAAndyZ/AGCI_Probe_Dataset).**
- **A subset of the data is already included in `probe_datasets` for reference.**

### Similarity Dataset
We also construct an auxiliary dataset to measure the cosine similarity between image features and their corresponding caption embeddings across different spatial positions. We randomly select 10,000 image-caption pairs from LAION and constructed 90,000 samples in total as well.

- **The dataset is publicly available on [🤗Hugging Face](https://huggingface.co/datasets/AAAndyZ/AGCI_Similarity_Dataset).**
- **A subset of the data is already included in `similarity_data` for reference.**

### Fine-tuning Dataset
We randomly sample 10,000 instruction-tuning examples from the LLaVA-v1.5 dataset to fine-tune LVLMs for adapting BaPA to general multi-modal downstream tasks.

- **The dataset can be found in `data/llava_v1_5_instruct_sample_10k_for_llava1.5.json`.**

### Downstream Benchmark

- **CRPE:** Download the dataset from [🤗Hugging Face](https://huggingface.co/datasets/OpenGVLab/CRPE).
- **HallusionBench:** Download the dataset from [🤗Hugging Face](https://huggingface.co/datasets/rayguan/HallusionBench).
- **ScienceQA:** We provide the formatted test data `llava_test_CQM-A.json` in `eval/ScienceQA/`. You should download and unzip test images (test.zip) from [Google Driver](https://drive.google.com/drive/folders/16kuhXdM-MOhYcFIyRj91WvnDnjnF-xHw).
- **MMMU-Pro:** Clone the repository from [Github](https://github.com/MMMU-Benchmark/MMMU) and download the dataset from [🤗Hugging Face](https://huggingface.co/datasets/MMMU/MMMU_Pro). We also provide the inference scripts in `eval/MMMU/mmmu-pro/infer`.
- **TextVQA**: Download the test data following [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
- **POPE**: Download the dataset from [🤗Hugging Face](https://huggingface.co/datasets/lmms-lab/POPE)


## 🧩 Models
You can download the following LVLMs directly from 🤗Hugging Face:

- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Gemma3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it)
- [InternVL-8B-hf](https://huggingface.co/OpenGVLab/InternVL3-8B-hf)
- [LLaVA-v1.6-Mistral-7B](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b)
- [LLaVA-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)
  
> [!IMPORTANT] 
> Before fine-tuning, please modify each model’s `config.json` file by adding the following parameter:

```json
"global_context_broadcast_lambda": your_lambda
```


## ⚙️ Usage

### Environment Setup
```bash
# First install LLaMA-Factory following https://github.com/hiyouga/LLaMA-Factory
# Then update transformers
cd transformers-4.57.1
pip install -e .
```

### Training with AGCI (LoRA Fine-tuning)
#### Image Preparation
Download the images of instruction tuning data provided by [LLaVA-v1.5](https://github.com/haotian-liu/LLaVA). 

After downloading all of them, organize the data as follows in `scripts/LLaVA/playground/data`
```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

#### LoRA Fine-tuning with 10K instruction tuning samples
1. Ensure that the model’s config.json includes:
    ```json
    "global_context_broadcast_lambda": your_lambda
    ```
2. Run LoRA-based fine-tuning and merge LoRA weights (More details please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).)
    ```bash
    # First install LLaMA-Factory, then
    llamafactory-cli train examples/train_lora/*.yaml
    llamafactory-cli export examples/merge_lora/*.yaml
    ```


### Evaluation on Probe Dataset
We provide the scripts of Qwen2.5-VL for reference.
```bash
cd scripts/Qwen2.5-VL
python eval_laion_bias_data.py --model_path path/to/model --eval_file path/to/probe/data --img_path path/to/probe/image --answer_file path/to/save/results --global_context_lambda your_lambda

# Generate heatmaps
cd scripts
pythom metrics.py --results_file path/to/saved/result --output_heatmap path/to/save/output/heatmap
```

### Evaluation on Perception Ability of Vision Encoder
We provide the scripts of Qwen2.5-VL for reference.
```bash
cd scripts/Qwen2.5-VL
python qwen_laion_bias_prob.py --model_path path/to/model --eval_file path/to/probe/data --img_path path/to/probe/image --output_dir path/to/save/results

# Generate heatmaps
cd scripts
pythom prob_results_position.py --prob_results_dir path/to/saved/result --output_path path/to/save/output/heatmaps
```

### Evaluation on Understanding Ability of Vision Encoder
We provide the scripts of Qwen2.5-VL for reference.
```bash
cd scripts/Qwen2.5-VL
python similarity --model_path path/to/model --eval_file path/to/similarity/data --img_path path/to/similarity/image --answer_file path/to/save/results

# Generate heatmaps
cd scripts
pythom metrics_similarity.py --results_file path/to/saved/results --output_heatmap path/to/save/output/heatmap
```

### Evaluation on ScienceQA, CRPE, HallusionBench, TextVQA and POPE
We provide the scripts of Qwen2.5-VL for reference.
```bash
cd scripts/Qwen2.5-VL
# First modify the arguments in eval_downstream.sh, then
sh scripts/eval_downstream.sh
```

### Evaluation on MMMU-Pro
We provide the scripts of Qwen2.5-VL for reference.
```bash
cd eval/MMMU/mmmu-pro/infer
python infer_qwen2.5vl.py --model path/to/model --dataset_variant standard (10 options)/standard (4 options) --dataset_repo_id path/to/MMMU_Pro_datasets --global_context_lambda your_lambda
```


## 🙏 Acknowledgements
We thank the developers of [LAION](https://laion.ai/), [LLaVA](https://huggingface.co/lmsys/vicuna-7b-v1.5), [Transformers](https://github.com/huggingface/transformers), [CCA](https://github.com/xing0047/cca-llava) and [Graphical Perception Evaluation](https://github.com/microsoft/lmm-graphical-perception) for open-sourcing their datasets, models and codes. This work builds upon their contributions.
