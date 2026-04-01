"""
Phi-4-mini-instruct on Modal with vLLM

Serves Microsoft Phi-4-mini-instruct (3.8B params, 128K context) as a
serverless OpenAI-compatible API on Modal, backed by vLLM.

Usage:
    modal deploy app.py     # deploy as a persistent endpoint
    modal run app.py        # quick smoke test (ephemeral)

Reference:
    https://modal.com/docs/examples/ministral3_inference
"""

import json
import os
import socket
import subprocess
from typing import Any

import aiohttp
import modal

# -- Timing helper --
# We express timeouts in minutes for readability, but Modal expects seconds.
MINUTES = 60  # seconds per minute

# Port the vLLM HTTP server listens on inside the container.
VLLM_PORT = 8000

# HuggingFace model identifier.
MODEL_NAME = "microsoft/Phi-4-mini-instruct"
MODEL_REVISION = "main"

# ---- Runtime configuration ------------------------------------------------
# All of these can be overridden by setting environment variables *before*
# running `modal deploy`.  Defaults work fine for most use-cases.
GPU_TYPE = os.environ.get("GPU_TYPE", "A100")
N_GPU = int(os.environ.get("N_GPU", "1"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
ENABLE_SNAPSHOTS = os.environ.get("ENABLE_SNAPSHOTS", "0") == "1"
SCALEDOWN_WINDOW = int(os.environ.get("SCALEDOWN_WINDOW", "10"))  # minutes
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "32"))

# ---- Modal app -------------------------------------------------------------
app = modal.App("phi4-mini-instruct-inference")

# ---- Container image -------------------------------------------------------
# Start from the official NVIDIA CUDA dev image, add Python 3.12, then install
# vLLM and its dependencies.  Modal provides the CUDA driver on the host, so
# we only need the toolkit/headers inside the image.
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.3",
    )
    .env(
        {
            # Enable high-performance Xet backend for faster weight downloads.
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)

# When GPU memory snapshots are enabled we need vLLM's dev/sleep mode and a
# single Torch Inductor compile thread (avoids snapshot compatibility issues).
if ENABLE_SNAPSHOTS:
    vllm_image = vllm_image.env(
        {
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )

# ---- Persistent volumes ----------------------------------------------------
# Model weights and vLLM compilation artefacts survive container restarts.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ---- In-container helpers ---------------------------------------------------
# These functions run inside the Modal container, so they import `requests`
# lazily via the image-scoped import block.
with vllm_image.imports():
    import requests as _requests


def _sleep(level: int = 1) -> None:
    """Put the vLLM server into sleep mode, offloading weights to CPU RAM."""
    _requests.post(
        f"http://localhost:{VLLM_PORT}/sleep?level={level}"
    ).raise_for_status()


def _wake_up() -> None:
    """Bring the vLLM server back from sleep mode."""
    _requests.post(f"http://localhost:{VLLM_PORT}/wake_up").raise_for_status()


def _wait_ready(proc: subprocess.Popen) -> None:
    """Block until the vLLM server is accepting TCP connections.

    Raises RuntimeError if the process dies before becoming ready.
    """
    while True:
        try:
            socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
            return
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited unexpectedly with code {proc.returncode}"
                )


def _warmup() -> None:
    """Fire a handful of throwaway requests so CUDA graphs get compiled.

    This is especially important when using memory snapshots -- the compiled
    graphs are captured into the snapshot and reused on subsequent cold starts.
    """
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        _requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=300,
        ).raise_for_status()


# ---- vLLM command builder --------------------------------------------------

def _build_vllm_cmd() -> list[str]:
    """Assemble the `vllm serve` CLI invocation."""
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        # Register both the full HF name and a short "llm" alias so callers
        # can use either in the `model` field of API requests.
        "--served-model-name", MODEL_NAME, "llm",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--tensor-parallel-size", str(N_GPU),
        "--trust-remote-code",
        "--max-model-len", str(MAX_MODEL_LEN),
    ]

    # Snapshot mode restricts concurrency and seq length so the GPU memory
    # layout is deterministic and can be snapshotted reliably.
    if ENABLE_SNAPSHOTS:
        cmd += [
            "--enable-sleep-mode",
            "--max-num-seqs", "2",
            "--max-num-batched-tokens", str(MAX_MODEL_LEN),
        ]

    return cmd


# ---- Server class -----------------------------------------------------------

@app.cls(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_WINDOW * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=ENABLE_SNAPSHOTS,
    **({"experimental_options": {"enable_gpu_snapshot": True}} if ENABLE_SNAPSHOTS else {}),
)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
class Phi4Server:
    """Wraps a vLLM subprocess and exposes it as a Modal web server.

    Lifecycle:
        1. start()  -- launch vLLM, wait for readiness, warm up CUDA graphs.
        2. serve()   -- proxy HTTP traffic to the local vLLM port.
        3. stop()    -- send SIGTERM to the vLLM process on container teardown.

    When ENABLE_SNAPSHOTS is true the start() method additionally puts the
    server to sleep after warmup so that a CPU+GPU memory snapshot can be
    captured.  The wake() method restores the server on subsequent launches.
    """

    @modal.enter(snap=ENABLE_SNAPSHOTS)
    def start(self):
        """Launch vLLM, wait until ready, and run warmup inference."""
        cmd = _build_vllm_cmd()
        print("Starting vLLM:", " ".join(cmd))
        self.vllm_proc = subprocess.Popen(cmd)
        _wait_ready(self.vllm_proc)
        _warmup()

        if ENABLE_SNAPSHOTS:
            _sleep()

    @modal.enter(snap=False)
    def wake(self):
        """Restore the vLLM server from sleep after a snapshot restore."""
        if ENABLE_SNAPSHOTS:
            _wake_up()
            _wait_ready(self.vllm_proc)

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        """Expose the OpenAI-compatible HTTP API to the internet."""
        pass

    @modal.exit()
    def stop(self):
        """Terminate the vLLM subprocess cleanly."""
        self.vllm_proc.terminate()


# ---- Local entrypoint -------------------------------------------------------
# `modal run app.py` spins up an ephemeral replica and runs this function on
# your local machine.  Handy for a quick sanity check after code changes.

@app.local_entrypoint()
async def test(test_timeout: int = 10 * MINUTES):
    """Run a basic health check and a single streamed inference request."""
    url = Phi4Server().serve.get_web_url()
    print(f"Server URL: {url}")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain what a neural network is in 2 sentences."},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        # -- Health check --
        print("Running health check ...")
        async with session.get(
            "/health", timeout=aiohttp.ClientTimeout(total=test_timeout - MINUTES)
        ) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed")

        # -- Streaming chat completion --
        print("\nChat completion (streaming):")
        await _stream_chat(session, "llm", messages, timeout=2 * MINUTES)


async def _stream_chat(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict],
    timeout: int,
) -> None:
    """Send a streaming /v1/chat/completions request and print tokens as they arrive."""
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 256,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with session.post(
        "/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]
            try:
                chunk = json.loads(line)
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
    print()  # newline after the streamed output
