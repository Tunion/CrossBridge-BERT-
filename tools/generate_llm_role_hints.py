from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import load_jsonl
from common.llm_role_hint_utils import save_llm_role_hints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate LLM role hints via an OpenAI-compatible chat endpoint.")
    p.add_argument("--input", type=str, default="outputs/results/llm_role_candidates.jsonl")
    p.add_argument("--output", type=str, default="outputs/results/llm_role_hints.jsonl")
    p.add_argument("--backend", type=str, choices=("api", "local"), default="api")
    p.add_argument("--api-base", type=str, default=os.environ.get("LLM_ROLE_API_BASE", ""))
    p.add_argument("--api-key", type=str, default=os.environ.get("LLM_ROLE_API_KEY", ""))
    p.add_argument("--model", type=str, default=os.environ.get("LLM_ROLE_MODEL", ""))
    p.add_argument("--local-model-dir", type=str, default="")
    p.add_argument("--local-device", type=str, choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--local-max-new-tokens", type=int, default=384)
    p.add_argument("--local-offload-dir", type=str, default="outputs/local_offload")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve_path(path_text: str, *, must_exist: bool) -> Path:
    raw = Path(str(path_text).strip())
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(ROOT / raw)
        candidates.append(raw)
    for cand in candidates:
        if not must_exist or cand.exists():
            return cand.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not resolve path: {path_text}. Tried: {tried}")


def call_chat(api_base: str, api_key: str, model: str, prompt: str, temperature: float) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    obj = json.loads(raw)
    content = obj["choices"][0]["message"]["content"]
    return parse_json_object(content)


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    marker = "FINAL_JSON="
    if marker in raw:
        raw = raw.split(marker, 1)[1].strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])
    raise ValueError(f"Could not parse JSON object from model output: {raw[:300]!r}")


def call_local_chat(
    model: Any,
    tokenizer: Any,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    import torch

    messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON extraction engine for Solidity role hints. "
                "Return one strict JSON object only. "
                "Do not output explanations, markdown, or chain-of-thought. "
                "Use numeric probabilities in [0,1] for auth_prob, external_prob, state_prob. "
                "Never output null. "
                "For bridge contracts, functions that approve, verify, execute, emit bridge events, or update processed/approved/nonces/message state often require non-zero role probabilities."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{prompt}\n\n"
                'Return exactly one JSON object in this format:\n'
                '{"auth_prob":0.0,"external_prob":0.0,"state_prob":0.0,"summary":"..."}'
            ),
        },
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    model_device = getattr(model, "device", None)
    if model_device is not None and str(model_device) != "meta":
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=bool(float(temperature) > 0.0),
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.eos_token_id,
            temperature=(float(temperature) if float(temperature) > 0.0 else None),
            top_p=(0.95 if float(temperature) > 0.0 else None),
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    content = tokenizer.decode(generated, skip_special_tokens=True)
    return parse_json_object(content)


def load_local_backend(model_dir: Path, device: str, offload_dir: Path) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    load_kwargs: Dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    torch_dtype = getattr(torch, "bfloat16", None)
    if device == "cpu":
        load_kwargs["device_map"] = "cpu"
    elif device == "cuda":
        load_kwargs["device_map"] = "cuda:0"
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
    else:
        offload_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs["device_map"] = "auto"
        load_kwargs["offload_folder"] = str(offload_dir)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input, must_exist=True)
    output_path = resolve_path(args.output, must_exist=False)
    rows = load_jsonl(input_path)
    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]
    if args.dry_run:
        print(
            "[generate_llm_role_hints] "
            f"dry_run items={len(rows)} backend={args.backend} model={args.model} api_base={args.api_base} "
            f"input={input_path} output={output_path}"
        )
        return
    local_model_dir = None
    if args.backend == "api":
        if not args.api_base or not args.model:
            raise RuntimeError("Missing --api-base or --model. Use an OpenAI-compatible local/remote endpoint.")
    else:
        if not str(args.local_model_dir).strip():
            raise RuntimeError("Missing --local-model-dir for local backend.")
        local_model_dir = resolve_path(args.local_model_dir, must_exist=True)
        if not local_model_dir.exists():
            raise RuntimeError(f"Local model dir not found: {local_model_dir}")
    hints: List[Dict[str, Any]] = []
    offload_dir = resolve_path(args.local_offload_dir, must_exist=False)
    local_backend = None
    if args.backend == "local":
        local_backend = load_local_backend(local_model_dir, str(args.local_device), offload_dir)
    for row in rows:
        if args.backend == "api":
            resp = call_chat(
                api_base=str(args.api_base),
                api_key=str(args.api_key),
                model=str(args.model),
                prompt=str(row.get("prompt", "")),
                temperature=float(args.temperature),
            )
            source = f"llm:{args.model}"
        else:
            model, tokenizer = local_backend
            resp = call_local_chat(
                model=model,
                tokenizer=tokenizer,
                prompt=str(row.get("prompt", "")),
                temperature=float(args.temperature),
                max_new_tokens=int(args.local_max_new_tokens),
            )
            source = f"local:{local_model_dir.name}"
        hints.append(
            {
                "relative_source_path": row.get("relative_source_path"),
                "function_name": row.get("function_name"),
                "auth_prob": float(resp.get("auth_prob", 0.0)),
                "external_prob": float(resp.get("external_prob", 0.0)),
                "state_prob": float(resp.get("state_prob", 0.0)),
                "summary": str(resp.get("summary", "") or ""),
                "source": source,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_llm_role_hints(output_path, hints)
    print(f"[generate_llm_role_hints] hints={len(hints)} output={output_path}")


if __name__ == "__main__":
    main()
