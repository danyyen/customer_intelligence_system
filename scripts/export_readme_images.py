"""Export selected rendered notebook figures for the project README."""

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_customer_segmentation.ipynb"
OUTPUT_DIR = ROOT / "images"

FIGURES = {
    119: "model_selection.png",
    157: "customer_activity_by_year.png",
    167: "final_segmentation_business_view.png",
    168: "final_segment_sizes.png",
}


def export_png(cell: dict, destination: Path) -> None:
    for output in cell.get("outputs", []):
        png = output.get("data", {}).get("image/png")
        if png:
            encoded = "".join(png) if isinstance(png, list) else png
            destination.write_bytes(base64.b64decode(encoded))
            return
    raise ValueError(f"No PNG output found for {destination.name}")


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(exist_ok=True)

    for cell_index, filename in FIGURES.items():
        export_png(notebook["cells"][cell_index], OUTPUT_DIR / filename)
        print(f"Exported {filename} from cell {cell_index}")


if __name__ == "__main__":
    main()
