import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from common import BASE_MODEL_ID, DEFAULT_GPTQ8_DIR
from prompts import QUESTIONS, SYSTEM_PROMPT, build_user_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize Arctic to GPTQ 8-bit.")
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_GPTQ8_DIR))
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--model-seqlen", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES override.")
    return parser.parse_args()


def build_calibration_texts(tokenizer):
    prompts = []
    for question in QUESTIONS:
        prompts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question)},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
        )
    return prompts


def main():
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration_texts = build_calibration_texts(tokenizer)
    quantization_config = GPTQConfig(
        bits=args.bits,
        tokenizer=args.model_id,
        dataset=calibration_texts,
        group_size=args.group_size,
        desc_act=False,
        sym=True,
        use_cuda_fp16=True,
        model_seqlen=args.model_seqlen,
        batch_size=args.batch_size,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)

    print(f"Saved Arctic GPTQ-{args.bits} model to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
