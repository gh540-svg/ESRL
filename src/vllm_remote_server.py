"""Serve one vLLM engine from the rollout H200 in a two-node allocation."""
from __future__ import annotations

import argparse
import json
import os
import traceback
from multiprocessing.managers import BaseManager
from pathlib import Path
from threading import Lock


class VLLMService:
    def __init__(self):
        self.engine = None
        self.lock = Lock()

    def initialize(self, config):
        with self.lock:
            if self.engine is not None:
                return {"ok": True, "reused": True}
            try:
                from vllm import LLM

                # These objects/classes are meaningful only in the server process.
                config = dict(config)
                self.engine = LLM(**config)
                return {"ok": True, "reused": False}
            except Exception as error:  # returned to the training node verbatim
                traceback.print_exc()
                return {"ok": False, "error": f"{type(error).__name__}: {error}"}

    def generate(self, prompts, sampling_payloads, lora_payloads):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampling = [SamplingParams(**dict(item)) for item in sampling_payloads]

        def build_lora(item):
            if item is None:
                return None
            kwargs = {}
            if item.get("long_lora_max_len") is not None:
                kwargs["long_lora_max_len"] = item["long_lora_max_len"]
            return LoRARequest(
                item["lora_name"], item["lora_int_id"], item["lora_path"],
                **kwargs,
            )

        if isinstance(lora_payloads, list):
            loras = [build_lora(item) for item in lora_payloads]
        else:
            loras = build_lora(lora_payloads)

        with self.lock:
            generated = self.engine.generate(
                list(prompts), sampling, lora_request=loras, use_tqdm=False
            )
        return [
            {
                "prompt_token_ids": list(result.prompt_token_ids),
                "token_ids": list(result.outputs[0].token_ids),
                "text": result.outputs[0].text,
            }
            for result in generated
        ]

    def collective_rpc(self, method, args):
        with self.lock:
            return self.engine.collective_rpc(method, args=tuple(args))


class RemoteManager(BaseManager):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--ready-file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # Match the single-node launcher: compiler artifacts on shared NFS have
    # previously produced stale-file-handle failures.  This process runs on
    # the rollout node, so create its caches from that node's local scratch.
    scratch = os.environ.get("SLURM_TMPDIR")
    if scratch:
        cache_root = Path(scratch)
        cache_dirs = {
            "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
            "TRITON_CACHE_DIR": cache_root / "triton",
            "VLLM_CACHE_ROOT": cache_root / "vllm",
        }
        for name, path in cache_dirs.items():
            os.environ[name] = str(path)
            path.mkdir(parents=True, exist_ok=True)
    service = VLLMService()
    RemoteManager.register("get_service", callable=lambda: service)
    manager = RemoteManager(
        address=(args.host, args.port), authkey=args.auth.encode()
    )
    server = manager.get_server()
    ready = Path(args.ready_file)
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text(json.dumps({
        "host": os.uname().nodename,
        "port": args.port,
        "pid": os.getpid(),
    }) + "\n", encoding="utf-8")
    print(
        f"[remote-vllm] listening on {os.uname().nodename}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
