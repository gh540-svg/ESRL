"""Small client adapter for running vLLM on a second Slurm node.

The training process keeps the public subset of ``vllm.LLM`` used by the
continual-learning trainers: ``generate`` and ``collective_rpc``.  Requests
are serialized through a ``multiprocessing.managers`` connection, while the
large model and KV cache remain exclusively on the rollout node.
"""
from __future__ import annotations

import os
import time
from multiprocessing.managers import BaseManager
from pathlib import Path
from types import SimpleNamespace


class _RemoteManager(BaseManager):
    pass


_RemoteManager.register("get_service")


def adapter_sync_parent():
    """Return storage visible to both trainer and remote rollout nodes."""
    if not os.environ.get("VLLM_REMOTE_HOST"):
        return os.environ.get("SLURM_TMPDIR") or None

    configured = os.environ.get("VLLM_SHARED_TMPDIR")
    if not configured:
        raise RuntimeError(
            "VLLM_SHARED_TMPDIR must be set when VLLM_REMOTE_HOST is set; "
            "node-local /tmp and SLURM_TMPDIR paths are invisible to the "
            "remote vLLM node"
        )
    shared = Path(configured).resolve()
    if not shared.is_dir():
        raise RuntimeError(f"VLLM_SHARED_TMPDIR is not a directory: {shared}")
    return str(shared)


def _sampling_payload(item):
    return {
        "temperature": float(item.temperature),
        "top_p": float(item.top_p),
        "max_tokens": int(item.max_tokens),
        "seed": None if item.seed is None else int(item.seed),
    }


def _lora_payload(item):
    if item is None:
        return None
    return {
        "lora_name": str(item.lora_name),
        "lora_int_id": int(item.lora_int_id),
        "lora_path": str(item.lora_path),
        "long_lora_max_len": getattr(item, "long_lora_max_len", None),
    }


class RemoteVLLMEngine:
    """Drop-in remote implementation for the vLLM calls used in this repo."""

    def __init__(self, **engine_config):
        host = os.environ.get("VLLM_REMOTE_HOST")
        port = int(os.environ.get("VLLM_REMOTE_PORT", "0"))
        auth = os.environ.get("VLLM_REMOTE_AUTH", "")
        if not host or not port or not auth:
            raise RuntimeError(
                "VLLM_REMOTE_HOST, VLLM_REMOTE_PORT and VLLM_REMOTE_AUTH "
                "must be set for the two-node rollout engine"
            )

        manager = _RemoteManager(address=(host, port), authkey=auth.encode())
        deadline = time.monotonic() + float(
            os.environ.get("VLLM_REMOTE_CONNECT_TIMEOUT", "300")
        )
        last_error = None
        while time.monotonic() < deadline:
            try:
                manager.connect()
                break
            except OSError as error:
                last_error = error
                time.sleep(2)
        else:
            raise RuntimeError(
                f"could not connect to remote vLLM at {host}:{port}"
            ) from last_error

        self._manager = manager
        self._service = manager.get_service()
        result = self._service.initialize(dict(engine_config))
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "remote vLLM initialization failed"))
        print(
            f"[remote-vllm] connected to {host}:{port}; "
            f"model={engine_config.get('model')}",
            flush=True,
        )

    def generate(self, prompts, sampling_params, lora_request=None, use_tqdm=False):
        del use_tqdm
        sampling = [_sampling_payload(item) for item in sampling_params]
        if isinstance(lora_request, (list, tuple)):
            loras = [_lora_payload(item) for item in lora_request]
        else:
            loras = _lora_payload(lora_request)
        payload = self._service.generate(list(prompts), sampling, loras)
        results = []
        for item in payload:
            output = SimpleNamespace(
                token_ids=list(item["token_ids"]),
                text=item["text"],
            )
            results.append(SimpleNamespace(
                prompt_token_ids=list(item["prompt_token_ids"]),
                outputs=[output],
            ))
        return results

    def collective_rpc(self, method, args=()):
        return self._service.collective_rpc(str(method), list(args))
