import json
import logging
from pathlib import Path
from time import perf_counter
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from datasets import Dataset, Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

jsonl_path = Path("Modality-Bias/data/vsr.jsonl")
new_dataset_path = Path("dataset_with_image/vsr")


def download_image(url):
    parsed_url = urlparse(url)
    if (
        parsed_url.scheme != "http"
        or parsed_url.hostname != "images.cocodataset.org"
    ):
        raise ValueError(f"Unexpected image host in URL: {url}")

    request = Request(url, headers={"User-Agent": "MixedSignalsDataset/1.0"})
    with urlopen(request, timeout=30) as response:
        image_bytes = response.read()

    if not image_bytes:
        raise ValueError(f"Downloaded an empty image from {url}")
    return image_bytes


combined_records = []
image_cache = {}
total_records = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
started_at = perf_counter()
logger.info("Processing %d VSR records from %s", total_records, jsonl_path)

with jsonl_path.open(encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        record = json.loads(line)
        image_url = record.get("image_link")
        if not image_url:
            raise ValueError(f"Missing image_link on JSONL line {line_number}")

        if image_url not in image_cache:
            try:
                image_cache[image_url] = download_image(image_url)
            except (OSError, TimeoutError, URLError) as error:
                logger.exception(
                    "Image download failed for record %s (line %d)",
                    record.get("id", line_number),
                    line_number,
                )
                raise RuntimeError(
                    f"Could not download image for record {record.get('id', line_number)} "
                    f"from {image_url}"
                ) from error
            logger.info(
                "Downloaded image %d (%s)",
                len(image_cache),
                record.get("image", "unknown filename"),
            )

        record["image_filename"] = record["image"]
        record["image"] = {"bytes": image_cache[image_url], "path": None}
        combined_records.append(record)
        if line_number % 50 == 0 or line_number == total_records:
            logger.info(
                "Prepared %d/%d records",
                line_number,
                total_records,
            )

combined_dataset = Dataset.from_list(combined_records)
combined_dataset = combined_dataset.cast_column("image", Image())
new_dataset_path.parent.mkdir(parents=True, exist_ok=True)
combined_dataset.save_to_disk(str(new_dataset_path))

logger.info(
    "Saved %d records with %d unique images to %s in %.1f seconds",
    len(combined_dataset),
    len(image_cache),
    new_dataset_path,
    perf_counter() - started_at,
)