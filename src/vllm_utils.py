"""Small vLLM request conversion helpers shared by the public trainers."""

import torch


def request_to_sequence(request_output, eos_token_id):
    prompt_ids = list(request_output.prompt_token_ids)
    completion_ids = list(request_output.outputs[0].token_ids)
    if not completion_ids:
        if eos_token_id is None:
            raise RuntimeError("vLLM returned an empty completion and no EOS token exists")
        completion_ids = [int(eos_token_id)]
    sequence = torch.tensor(
        prompt_ids + completion_ids, dtype=torch.long
    ).unsqueeze(0)
    return sequence, len(prompt_ids)
