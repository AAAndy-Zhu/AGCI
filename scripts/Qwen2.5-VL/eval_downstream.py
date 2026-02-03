import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from transformers.generation import GenerationConfig
import torch
import json
from tqdm import tqdm
import re
import random
import argparse
import pandas as pd
from PIL import Image
from io import BytesIO
torch.manual_seed(1234)


def eval_downstream(args):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path)

    datasets = ['scienceqa', 'hallusionbench', 'crpe', 'textqa', 'pope']
    model.global_context_broadcast_lambda = args.global_context_lambda 

    print("Global Context Lambda:", model.global_context_broadcast_lambda)

    for dataset in datasets:
        print("Tested Dataset:", dataset)

        if dataset == 'scienceqa':
            eval_file_path = args.eval_file_path_scienceqa
            image_path = args.image_path_scienceqa
            answers_file = args.answers_file_scienceqa
        elif dataset == 'hallusionbench':
            eval_file_path = args.eval_file_path_hallusionbench
            image_path = args.image_path_hallusionbench
            answers_file = args.answers_file_hallusionbench
        elif dataset == 'crpe':
            eval_file_path = args.eval_file_path_crpe
            image_path = args.image_path_crpe
            answers_file = args.answers_file_crpe
        elif dataset == 'textqa':
            eval_file_path = args.eval_file_path_textqa
            image_path = args.image_path_textqa
            answers_file = args.answers_file_textqa
        elif dataset == 'pope':
            eval_file_path = args.eval_file_path_pope
            df = pd.read_parquet(eval_file_path)
            answers_file = args.answers_file_pope
        else:
            raise NotImplementedError('Not Implemented Dataset.')

        if dataset in ['scienceqa', 'hallusionbench']:
            test_data = json.load(open(eval_file_path))
        elif dataset in ['pope']:
            test_data = []
            for index, row in df.iterrows():
                item = {
                    'question_id': row['question_id'],
                    'question': row['question'],
                    'image_source': row['image_source'],
                    'image': row['image']['bytes'],
                    'answer': row['answer'],
                    'category': row['category'],
                }
                test_data.append(item)
        else:
            test_data = [json.loads(line) for line in open(eval_file_path)]

        ans_file = open(answers_file, "w")
        for data in tqdm(test_data):
            if dataset == 'scienceqa':
                if 'image' not in data:
                    continue
                qs = data['conversations'][0]['value'] + "\nAnswer with the option's letter from the given choices directly."
                image_file = os.path.join(image_path, data['image'])
            elif dataset == ['textvqa', 'crpe']:
                qs = data['text']
                image_file = os.path.join(image_path, data['image'])
            elif dataset == 'hallusionbench':
                if data['visual_input'] == "0":
                    continue
                qs = data['question'] + "\nThe answer should only contain 'Yes' or 'No', without reasoning process."
                image_file = os.path.join(image_path, data['filename'])
            elif dataset == 'pope':
                qs = data['question'] + "\nAnswer the question using a single word or phrase."
                image_file = Image.open(BytesIO(data['image'])).convert('RGB')
            else:
                raise NotImplementedError('Not Implemented Dataset.')
            # print(qs)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_file,
                        },
                        {"type": "text", "text": qs},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

            if dataset == 'textvqa':
                ans_file.write(json.dumps({
                    "question_id": data['question_id'],
                    "image": data['image'],
                    "prompt": qs,
                    "text": output_text,
                }) + "\n")
                ans_file.flush()
            elif dataset == 'pope':
                ans_file.write(json.dumps({
                    'question_id': data['question_id'],
                    'question': data['question'],
                    'image_source': data['image_source'],
                    'answer': data['answer'],
                    'prediction': output_text,
                    'prompt': qs,
                    'category': data['category'],
                }) + "\n")
                ans_file.flush()
            else:
                data['prediction'] = output_text

                ans_file.write(json.dumps(data) + "\n")
                ans_file.flush()
        ans_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/Qwen2.5-VL', help='model path')

    parser.add_argument('--eval_file_path_scienceqa', type=str, default='/path/to/scienceqa/test.json', help='evaluation file path for scienceqa')
    parser.add_argument('--image_path_scienceqa', type=str, default='/path/to/scienceqa/images', help='image file path for scienceqa')
    parser.add_argument('--answers_file_scienceqa', type=str, default='./results_downstream/scienceqa_answers.json', help='answers file for scienceqa')

    parser.add_argument('--eval_file_path_hallusionbench', type=str, default='/path/to/hallusionbench/test.json', help='evaluation file path for hallusionbench')
    parser.add_argument('--image_path_hallusionbench', type=str, default='/path/to/hallusionbench/images', help='image file path for hallusionbench')
    parser.add_argument('--answers_file_hallusionbench', type=str, default='./results_downstream/hallusionbench_answers.json', help='answers file for hallusionbench')

    parser.add_argument('--eval_file_path_crpe', type=str, default='/path/to/crpe/test.jsonl', help='evaluation file path for crpe')
    parser.add_argument('--image_path_crpe', type=str, default='/path/to/crpe/images', help='image file path for crpe')
    parser.add_argument('--answers_file_crpe', type=str, default='./results_downstream/crpe_answers.json', help='answers file for crpe')

    parser.add_argument('--eval_file_path_textqa', type=str, default='/path/to/textqa/test.jsonl', help='evaluation file path for textqa')
    parser.add_argument('--image_path_textqa', type=str, default='/path/to/textqa/images', help='image file path for textqa')
    parser.add_argument('--answers_file_textqa', type=str, default='./results_downstream/textqa_answers.json', help='answers file for textqa')

    parser.add_argument('--eval_file_path_pope', type=str, default='/path/to/pope/test.jsonl', help='evaluation file path for pope')
    parser.add_argument('--answers_file_pope', type=str, default='./results_downstream/pope_answers.json', help='answers file for pope')

    parser.add_argument('--global_context_lambda', type=float, default=0.0)

    args = parser.parse_args()

    eval_downstream(args)