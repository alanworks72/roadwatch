import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
URL = "https://datasets-cvlab.epfl.ch/2019-road-anomaly/RoadAnomaly.zip"
SHA256 = "624836882fb7ef2473c76257f3d6418caebedfc21bd20a360057a8b32bf33ced" ## RoadAnomaly.zip Hash


def main():
    destination = ROOT / "data" / "road-anomaly"
    destination.mkdir(parents=True, exist_ok=True)

    archive = destination / "RoadAnomaly.zip"

    if not archive.exists():
        temporary = archive.with_suffix(".part")

        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)

        temporary.replace(archive)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest() ## 파일 변조 검사

    if digest != SHA256:
        raise ValueError("Dataset checksum differs from the verified archive; inspect before extracting")
    
    with zipfile.ZipFile(archive) as source:
        bad = source.testzip()

        if bad:
            raise ValueError(f"Corrupt archive member: {bad}")
        
        for member in source.infolist():
            target = (destination / member.filename).resolve()

            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Archive contains a path outside the dataset directory")
            
        source.extractall(destination)

    images = sorted(destination.rglob("*.webp"))

    if not images:
        raise ValueError("No WebP images found; inspect the upstream archive format")
    
    records = []
    for path in images:
        mask = path.parent / (path.stem + ".labels") / "labels_semantic.png"

        if not mask.is_file():
            raise ValueError(f"Missing mask: {mask}")
        
        with Image.open(path) as image, Image.open(mask) as labels:
            if image.size != labels.size:
                raise ValueError(f"Image/mask size mismatch: {path.name}")
            
            values = sorted(set(labels.get_flattened_data())) ## 이미지 비교를 위한 flatten, set, sort
            records.append({"image": str(path.relative_to(ROOT)), "mask": str(mask.relative_to(ROOT)),
                            "size": list(image.size), "label_values": values})
            
    report = {"source": URL, "sha256": digest,
              "image_count": len(records), "records": records,
              "scope": "Local research only; no redistribution permission inferred."}
    
    output = ROOT / "outputs" / "road-anomaly-audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"image_count": len(records), "sha256": report["sha256"],
                      "report": str(output.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
