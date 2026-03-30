import argparse
import os
import re
from pathlib import Path

from common import DEFAULT_GPTQ8_DIR, DEFAULT_GPTQ8_REPO_ID, load_hf_token
from prompts import QUESTIONS, SYSTEM_PROMPT, build_user_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Test an Arctic model with the standard Text2SQL prompt set.")
    parser.add_argument("--model", default=DEFAULT_GPTQ8_REPO_ID, help="Local path or Hugging Face repo.")
    parser.add_argument("--tokenizer", help="Tokenizer path or repo. Defaults to --model.")
    parser.add_argument("--from-hub", action="store_true", help="Download the model repo to a local cache first.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_GPTQ8_DIR.parent / "arctic-test-cache"))
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--quantization", default="gptq_marlin")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--question-index", type=int, help="1-based question index to run instead of all questions.")
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES override.")
    return parser.parse_args()


def resolve_model_source(args, snapshot_download):
    model_source = args.model
    tokenizer_source = args.tokenizer or args.model
    if not args.from_hub:
        return model_source, tokenizer_source

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    token = load_hf_token()
    model_source = snapshot_download(
        repo_id=args.model,
        local_dir=str(cache_dir),
        token=token,
    )
    tokenizer_source = model_source if args.tokenizer is None else args.tokenizer
    return model_source, tokenizer_source


def extract_sql(text: str) -> str:
    match = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if match:
        return f"```sql\n{match.group(1).strip()}\n```"
    return text


def main():
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_source, tokenizer_source = resolve_model_source(args, snapshot_download)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        fix_mistral_regex=True,
        trust_remote_code=True,
    )

    selected_questions = QUESTIONS
    if args.question_index:
        if args.question_index < 1 or args.question_index > len(QUESTIONS):
            raise ValueError(f"--question-index must be between 1 and {len(QUESTIONS)}.")
        selected_questions = [QUESTIONS[args.question_index - 1]]

    prompts = []
    for question in selected_questions:
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

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens, n=1)
    llm = LLM(
        model=model_source,
        tokenizer=tokenizer_source,
        load_format=args.load_format,
        quantization=args.quantization,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        hf_token=load_hf_token(),
        trust_remote_code=True,
    )

    outputs = llm.generate(prompts, sampling_params)
    for index, output in enumerate(outputs, start=1):
        print(f"Question {index}:")
        print(selected_questions[index - 1])
        print(extract_sql(output.outputs[0].text.strip()))
        print()


if __name__ == "__main__":
    main()
