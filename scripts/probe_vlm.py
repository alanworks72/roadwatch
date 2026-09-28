import argparse
import json
from pathlib import Path
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PROMPT = (
    "Inspect this road scene. Briefly describe visible objects or surface damage that may "
    "obstruct vehicle passage. Separate direct visual evidence from uncertainty. "
    "Do not infer depth, duration, exact location, or hidden hazards. "
    "Suggest a review priority (low, medium, high, or uncertain) and explain why. "
    "If evidence is insufficient, request human review."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "vlm-probe.json")

    args = parser.parse_args()

    if args.max_side < 32 or args.max_new_tokens < 1:
        parser.error("max-side must be >= 32 and max-new-tokens must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this feasibility check")
    
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    image.thumbnail((args.max_side, args.max_side))

    started = time.perf_counter()

    print(f"Loading {MODEL}; first run downloads about 4.3 GB of weights.", file=sys.stderr, flush=True)

    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    ).eval() ## 지수 비트가 FP32와 같은 BF16 타입 선택 / Scaled Dot Product Attention

    torch.cuda.synchronize()

    load_seconds = time.perf_counter() - started

    print("Model loaded; running one image on CUDA.", file=sys.stderr, flush=True)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": args.prompt}]}] ## Single-turn
    
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to("cuda") ## inputs: input_ids, attention_mask, pixel_values, image_grid_thw
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False) ## greedy selection

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    tokens = outputs[:, inputs["input_ids"].shape[1]:] ## input token 제거
    report = {"model": MODEL, "revision": REVISION, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "image_name": args.image.name,
              "input_size": list(image.size), "prompt": args.prompt,
              "image_grid_thw": inputs["image_grid_thw"].tolist(),
              "pixel_tensor_shape": list(inputs["pixel_values"].shape),
              "dtype": "bfloat16", "attention": "sdpa", "max_new_tokens": args.max_new_tokens,
              "generated_tokens": tokens.shape[1], "load_seconds_including_download": load_seconds,
              "preprocess_and_generate_seconds": elapsed,
              "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / 2**20,
              "response": processor.batch_decode(tokens, skip_special_tokens=True)[0],
              "scope": "Single cold inference; no quality or real-time performance claim."}
    
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
