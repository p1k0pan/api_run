#!/usr/bin/env python3
"""Run OpenAI-compatible commercial LVLM APIs on VIDA rebuttal splits.

The API credential file defaults to /mnt/workspace/xintong/api_key.txt and must
contain two lines:
  line 1: API key
  line 2: base URL
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from tqdm import tqdm


DATASET_FILES = {
    "base": ("VIDA-Base-Test", "VIDA-Base-Test.json", "ambi_normal_test_rest.json"),
    "sent": ("VIDA-Sent", "VIDA-Sent.json", "mma_final.json"),
    "colln": ("VIDA-CollN", "VIDA-CollN.json", "sp_final_filter_clean.json"),
}

SYSTEM_PROMPT = (
    "You are a translation expert. Your task is to translate the English sentence into Chinese.  \n"
    "Note: ONLY return the Chinese translation without any additional text or explanations."
)
USER_TEMPLATE = "Please translate the following English sentence into Chinese: {en}"
THINKING_HINT = (
    "Carefully use the image to resolve any ambiguity in the English sentence. "
    "Do not output your reasoning. Return only the final Chinese translation."
)


def read_api_key_file(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"{path} must contain API_KEY on line 1 and BASE_URL on line 2.")
    api_key = lines[0].strip()
    base_url = lines[1].strip().rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return api_key, base_url


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_image_path(data_root: Path, item: dict[str, Any], image_root: Path | None = None) -> Path:
    image = str(item.get("image", "")).strip()
    candidates = [
        data_root / image,
        data_root / "images" / Path(image).name,
        data_root / "MMA" / Path(image).name,
        data_root / "vida_sent" / Path(image).name,
    ]
    if image_root is not None:
        candidates = [
            image_root / image,
            image_root / Path(image).name,
            image_root / "images" / Path(image).name,
            image_root / "MMA" / Path(image).name,
            image_root / "vida_sent" / Path(image).name,
        ] + candidates
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Image not found for idx={item.get('idx')}: {image}")


def choose_random_images(items: list[dict[str, Any]], data_root: Path, image_root: Path | None, seed: int) -> list[Path]:
    image_paths = [resolve_image_path(data_root, item, image_root) for item in items]
    rng = random.Random(seed)
    assigned = []
    n = len(image_paths)
    for i, path in enumerate(image_paths):
        if n <= 1:
            assigned.append(path)
            continue
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        assigned.append(image_paths[j])
    return assigned


def image_to_data_url(path: Path, max_side: int, jpeg_quality: int, raw_image: bool) -> str:
    if raw_image:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"

    img = Image.open(path).convert("RGB")
    long_side = max(img.width, img.height)
    if max_side > 0 and long_side > max_side:
        scale = max_side / long_side
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def build_messages(
    en: str,
    image_path: Path | None,
    condition: str,
    thinking_mode: str,
    image_max_side: int,
    jpeg_quality: int,
    raw_image: bool,
) -> list[dict[str, Any]]:
    system_prompt = SYSTEM_PROMPT
    user_text = USER_TEMPLATE.format(en=en.strip())
    # if thinking_mode == "thinking":
    #     user_text = THINKING_HINT + "\n\n" + user_text

    if condition == "text_only":
        content: Any = user_text
    else:
        assert image_path is not None
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(
                        image_path,
                        max_side=image_max_side,
                        jpeg_quality=jpeg_quality,
                        raw_image=raw_image,
                    )
                },
            },
            {"type": "text", "text": user_text},
        ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def clean_translation(text: str) -> str:
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "<answer>" in text and "</answer>" in text:
        text = text.split("<answer>")[-1].split("</answer>")[0].strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"^(中文翻译|翻译|译文|答案)\s*[:：]\s*", "", text).strip()
    if len(text.splitlines()) > 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            text = lines[-1]
    return text.strip(" \t\r\n\"“”")


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for item in load_json(path):
        keys.add((str(item.get("idx", "")), str(item.get("image", "")), str(item.get("en", ""))))
    return keys


def call_api(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body=extra_body or None,
    )
    message = response.choices[0].message
    message_dump = message.model_dump() if hasattr(message, "model_dump") else dict(message)
    reasoning_parts = []
    for key in ("reasoning_content", "reasoning", "reasoning_details", "reasoning_summary"):
        value = message_dump.get(key)
        if value:
            reasoning_parts.append(value)
    return {
        "content": message.content or "",
        "reasoning_content": "\n".join(str(x) for x in reasoning_parts),
        "message": message_dump,
    }


def generate_dataset(
    args: argparse.Namespace,
    client: OpenAI,
    dataset_label: str,
    input_path: Path,
    output_name: str,
    condition: str,
    thinking_mode: str,
) -> Path:
    items = load_json(input_path)
    if args.limit > 0:
        items = items[: args.limit]

    run_name = args.run_name if thinking_mode == "no_thinking" else f"{args.run_name}_{thinking_mode}"
    out_dir = args.output_root / run_name / condition
    out_path = out_dir / output_name
    old_items = load_json(out_path) if out_path.exists() and args.resume else []
    done = existing_keys(out_path) if args.resume else set()

    if condition == "correct_image":
        image_paths: list[Path | None] = [resolve_image_path(args.data_root, item, args.image_root) for item in items]
    elif condition == "random_image":
        image_paths = choose_random_images(items, args.data_root, args.image_root, args.seed + hash(str(input_path)) % 100000)
    elif condition == "text_only":
        image_paths = [None for _ in items]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    jobs = []
    for pos, item in enumerate(items):
        key = (str(item.get("idx", "")), str(item.get("image", "")), str(item.get("en", "")))
        if key in done:
            continue
        messages = build_messages(
            str(item.get("en", "")),
            image_paths[pos],
            condition,
            thinking_mode,
            args.image_max_side,
            args.jpeg_quality,
            args.raw_image,
        )
        jobs.append((pos, item, messages, image_paths[pos]))

    results: list[dict[str, Any] | None] = [None] * len(jobs)
    failures: list[dict[str, Any]] = []
    lock = threading.Lock()
    completed = 0

    order = {
        (str(item.get("idx", "")), str(item.get("image", "")), str(item.get("en", ""))): i
        for i, item in enumerate(items)
    }

    def merged() -> list[dict[str, Any]]:
        data = list(old_items) + [item for item in results if item is not None]
        data.sort(
            key=lambda x: order.get(
                (str(x.get("idx", "")), str(x.get("image", "")), str(x.get("en", ""))),
                10**9,
            )
        )
        return data

    def save_progress() -> None:
        write_json(out_path, merged())
        if failures:
            write_json(out_path.with_name(out_path.stem + "_failures.json"), failures)

    def run_one(job: tuple[int, dict[str, Any], list[dict[str, Any]], Path | None]) -> tuple[int, dict[str, Any]]:
        pos, item, messages, used_image = job
        last_error: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                api_response = call_api(
                    client,
                    args.model,
                    messages,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                    args.timeout,
                    args.extra_body_json,
                )
                raw = api_response["content"]
                result = clean_translation(raw)
                out = {
                    "idx": item.get("idx"),
                    "image": item.get("image"),
                    "en": item.get("en"),
                    "standard_zh": item.get("standard_zh", ""),
                    "fg_zh": item.get("fine_grained_zh", "") or item.get("fg_zh", "") or "",
                    "result": result,
                    "raw_output": raw,
                    "reasoning_content": api_response["reasoning_content"],
                    "raw_message": api_response["message"],
                    "api_model": args.model,
                    "api_condition": condition,
                    "api_dataset": dataset_label,
                    "api_thinking_mode": thinking_mode,
                    "used_image": str(used_image) if used_image else "",
                }
                return pos, out
            except Exception as exc:
                last_error = exc
                if attempt >= args.retries:
                    raise
                time.sleep(args.retry_sleep * (attempt + 1))
        raise RuntimeError(last_error)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run_one, job): i for i, job in enumerate(jobs)}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc=f"{run_name}/{condition}/{dataset_label}"):
            slot = future_map[future]
            try:
                _, out_item = future.result()
                results[slot] = out_item
            except Exception as exc:
                _, item, _, used_image = jobs[slot]
                failure = {
                    "idx": item.get("idx"),
                    "image": item.get("image"),
                    "en": item.get("en"),
                    "condition": condition,
                    "thinking_mode": thinking_mode,
                    "used_image": str(used_image) if used_image else "",
                    "error": repr(exc),
                }
                failures.append(failure)
                tqdm.write(f"[WARN] failed item: {failure}")
                if args.fail_fast:
                    save_progress()
                    raise
            with lock:
                completed += 1
                if args.save_every > 0 and completed % args.save_every == 0:
                    save_progress()

    save_progress()
    manifest = {
        "run_name": run_name,
        "model": args.model,
        "dataset": dataset_label,
        "condition": condition,
        "thinking_mode": thinking_mode,
        "items": len(items),
        "generated_items": len(merged()),
        "failed_items": len(failures),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "data_root": str(args.data_root),
        "output_path": str(out_path),
    }
    write_json(out_path.with_name(out_path.stem + "_manifest.json"), manifest)
    print(f"Wrote {out_path}")
    return out_path


def dataset_jobs_from_keys(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    jobs = []
    for dataset_key in parse_csv(args.datasets):
        dataset_label, input_name, output_name = DATASET_FILES[dataset_key]
        jobs.append((dataset_label, args.data_root / input_name, output_name))
    return jobs


def dataset_jobs_from_files(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    files = [Path(x).expanduser() for x in parse_csv(args.json_files)]
    output_names = parse_csv(args.output_names) if args.output_names else []
    labels = parse_csv(args.dataset_labels) if args.dataset_labels else []
    if output_names and len(output_names) != len(files):
        raise SystemExit("--output-names must have the same number of entries as --json-files.")
    if labels and len(labels) != len(files):
        raise SystemExit("--dataset-labels must have the same number of entries as --json-files.")
    jobs = []
    for i, path in enumerate(files):
        label = labels[i] if labels else path.stem
        output_name = output_names[i] if output_names else path.name
        jobs.append((label, path, output_name))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-file", type=Path, default=Path("/mnt/workspace/xintong/api_key.txt"))
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Folder containing manually downloaded images. Required for correct_image/random_image.",
    )
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", default="base,sent,colln")
    parser.add_argument(
        "--json-files",
        default="",
        help="Comma-separated JSON files to run. If set, --datasets is ignored.",
    )
    parser.add_argument(
        "--output-names",
        default="",
        help="Comma-separated output filenames for --json-files. Defaults to original filenames.",
    )
    parser.add_argument(
        "--dataset-labels",
        default="",
        help="Comma-separated labels for --json-files. Defaults to JSON stems.",
    )
    parser.add_argument("--conditions", default="correct_image,text_only,random_image")
    parser.add_argument("--thinking-modes", default="thinking")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--image-max-side", type=int, default=1536)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--raw-image", action="store_true")
    parser.add_argument("--extra-body", default="", help="Optional JSON object passed as extra_body.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    api_key, base_url = read_api_key_file(args.api_key_file)
    args.extra_body_json = json.loads(args.extra_body) if args.extra_body else None
    client = OpenAI(api_key=api_key, base_url=base_url)

    conditions = parse_csv(args.conditions)
    thinking_modes = parse_csv(args.thinking_modes)
    if args.image_root is not None:
        args.image_root = args.image_root.expanduser()
    if args.json_files:
        dataset_jobs = dataset_jobs_from_files(args)
    else:
        datasets = parse_csv(args.datasets)
        unknown_datasets = sorted(set(datasets) - set(DATASET_FILES))
        if unknown_datasets:
            raise SystemExit(f"Unknown datasets: {unknown_datasets}. Use base,sent,colln.")
        dataset_jobs = dataset_jobs_from_keys(args)
    unknown_conditions = sorted(set(conditions) - {"correct_image", "text_only", "random_image"})
    if unknown_conditions:
        raise SystemExit(f"Unknown conditions: {unknown_conditions}.")
    if any(condition in {"correct_image", "random_image"} for condition in conditions) and args.image_root is None:
        raise SystemExit(
            "--image-root is required when conditions include correct_image or random_image. "
            "Use --image-root /path/to/downloaded_images."
        )
    unknown_modes = sorted(set(thinking_modes) - {"no_thinking", "thinking"})
    if unknown_modes:
        raise SystemExit(f"Unknown thinking modes: {unknown_modes}.")

    for mode in thinking_modes:
        for condition in conditions:
            for dataset_label, input_path, output_name in dataset_jobs:
                generate_dataset(args, client, dataset_label, input_path, output_name, condition, mode)


if __name__ == "__main__":
    main()
