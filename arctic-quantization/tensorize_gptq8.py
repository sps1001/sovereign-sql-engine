import argparse
import os
import shutil
from pathlib import Path

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig,
    tensorize_vllm_model,
)

from common import DEFAULT_GPTQ8_DIR, DEFAULT_GPTQ8_REPO_ID, DEFAULT_TENSORIZED_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Tensorize a local GPTQ-8 Arctic model.")
    parser.add_argument("--source-dir", default=str(DEFAULT_GPTQ8_DIR))
    parser.add_argument("--served-model-name", default=DEFAULT_GPTQ8_REPO_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_TENSORIZED_DIR))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--cuda-visible-devices", help="Optional CUDA_VISIBLE_DEVICES override.")
    return parser.parse_args()


def copy_model_artifacts(source_dir: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.name.startswith("model-"):
            continue
        if path.name == "model.tensors":
            continue
        if path.is_file():
            shutil.copy2(path, target_dir / path.name)


def main():
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    source_dir = Path(args.source_dir)
    target_dir = Path(args.output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"GPTQ source model not found at {source_dir.resolve()}.")

    engine_args = EngineArgs(
        model=str(source_dir),
        tokenizer=str(source_dir),
        served_model_name=args.served_model_name,
        quantization="gptq_marlin",
        dtype="float16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    tensorizer_config = TensorizerConfig(tensorizer_dir=str(target_dir))
    tensorize_vllm_model(
        engine_args=engine_args,
        tensorizer_config=tensorizer_config,
        generate_keyfile=False,
    )
    copy_model_artifacts(source_dir, target_dir)

    print(f"Tensorized GPTQ model saved to {target_dir.resolve()}")


if __name__ == "__main__":
    main()
