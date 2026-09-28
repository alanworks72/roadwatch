import argparse
import json
from pathlib import Path
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


MODEL = "IDEA-Research/grounding-dino-tiny"
REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71" ## 개발 환경 재현을 위한 버전 정보


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--prompt", default="an obstacle on the road. an animal. a rock. a traffic cone. a pothole.")
    parser.add_argument("--threshold", type=float, default=0.25)

    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this feasibility check")
    
    with Image.open(args.image) as source:
        image = source.convert("RGB")

    print(f"Loading {MODEL}; first run downloads model weights.", file=sys.stderr, flush=True)

    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION) ## image_processor, tokenizer
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL, revision=REVISION).to("cuda").eval() ## Grounding Dino tiny: Image = Swin / Text = BERT
    
    torch.cuda.synchronize()
    
    print("Model loaded; running one image on CUDA.", file=sys.stderr, flush=True)
    
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    inputs = processor(images=image, text=args.prompt, return_tensors="pt").to("cuda") ## resize, rescale, normalize, padding
    
    with torch.inference_mode():
        raw = model(**inputs)

    result = processor.post_process_grounded_object_detection(raw, inputs.input_ids, threshold=args.threshold, text_threshold=0.25, target_sizes=[image.size[::-1]])[0]
    
    torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    labels = result["text_labels"]
    report = {"model": MODEL, "revision": REVISION, "prompt": args.prompt,
              "threshold": args.threshold, "text_threshold": 0.25,
              "image_name": args.image.name, "original_size": list(image.size),
              "tensor_shape": list(inputs.pixel_values.shape),
              "preprocess_infer_postprocess_seconds": elapsed,
              "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
              "detections": [{"label": label, "score": float(score), "box_xyxy": box.tolist()}
                             for label, score, box in zip(labels, result["scores"], result["boxes"])],
              "scope": "Single cold inference; prompts are supplied concepts, not proof of unknown recall."}
    
    output = Path(__file__).resolve().parents[1] / "outputs" / "detector-probe.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
