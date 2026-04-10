"""Quantize Arctic model using GPTQ 8-bit quantization.

This script performs GPTQ-8 quantization on the Arctic model using calibration questions
from the Text2SQL task. GPTQ quantization provides better performance than BitsAndBytes
with similar memory savings.
"""
import argparse
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from common import BASE_MODEL_ID, DEFAULT_GPTQ8_DIR, setup_logging
from prompts import QUESTIONS, SYSTEM_PROMPT, build_user_prompt


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for GPTQ quantization.
    
    Returns:
        argparse.Namespace: Arguments for model path, output, bit depth, group size,
                          sequence length, batch size, and GPU settings.
    """
    parser = argparse.ArgumentParser(description="Quantize Arctic to GPTQ 8-bit.")
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_GPTQ8_DIR))
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--model-seqlen", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--log-level", default="INFO", help="Logging level for debugging.")
    return parser.parse_args()


def build_calibration_texts(tokenizer):
    """Build calibration texts from test questions and system prompt.
    
    Uses Text2SQL test questions formatted as chat templates for GPTQ calibration.
    
    Args:
        tokenizer: Loaded tokenizer for the model.
        
    Returns:
        list: Formatted prompt strings for calibration.
    """
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
    """Main entry point: quantize and save Arctic model in GPTQ 8-bit format.
    
    Builds calibration texts, configures GPTQ quantization, loads the model,
    and saves the quantized result with tokenizer.
    """
    args = parse_args()
    setup_logging(args.log_level)
    logger.info("Starting GPTQ quantization for model %s", args.model_id)
    logger.debug(
        "Quantization config: bits=%s group_size=%s model_seqlen=%s batch_size=%s output_dir=%s",
        args.bits,
        args.group_size,
        args.model_seqlen,
        args.batch_size,
        args.output_dir,
    )

    # Set GPU device visibility if specified
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        logger.info("Using CUDA_VISIBLE_DEVICES=%s", args.cuda_visible_devices)

    # Load tokenizer and set padding token if needed
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        logger.debug("Tokenizer has no pad token; using eos token")
        tokenizer.pad_token = tokenizer.eos_token

    # Build calibration dataset and create GPTQ configuration
    logger.info("Building calibration dataset from %d questions", len(QUESTIONS))
    calibration_texts = build_calibration_texts(tokenizer)
    logger.debug("Calibration dataset size: %d", len(calibration_texts))
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

    # Load model with GPTQ quantization configuration
    logger.info("Loading model with GPTQ quantization")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Save quantized model and tokenizer
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving tokenizer and model to %s", output_dir.resolve())
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)

    logger.info("Saved Arctic GPTQ-%s model to %s", args.bits, output_dir.resolve())


if __name__ == "__main__":
    main()
