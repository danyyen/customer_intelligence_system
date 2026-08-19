"""Export selected rendered notebook figures for the project README.

Figures are looked up by cell ID, not position. The notebook gets edited
often enough (cells added/removed) that a hardcoded index silently points
at the wrong cell -- or an out-of-range one -- the next time it changes;
the ID is the one thing about a cell that's actually stable.
"""

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_customer_segmentation.ipynb"
OUTPUT_DIR = ROOT / "images"

# cell_id -> output filename. Find a cell's id in the notebook JSON
# (VS Code / Jupyter show it in cell metadata) if these ever need updating.
FIGURES = {
    "71e547ce": "model_selection.png",                      # "Elbow Method: Inertia vs. k"
    "f4cc5907": "customer_activity_by_year.png",             # "Customer Activity Across Dataset Years by Segment"
    "5b707ef0": "final_segmentation_business_view.png",      # "Final Segmentation: Customers by Recency, Value and Purchase Frequency"
    "15b6e9f2": "final_segment_sizes.png",                   # "Final Segment Sizes"
}


def export_png(cell: dict, destination: Path) -> None:
    for output in cell.get("outputs", []):
        png = output.get("data", {}).get("image/png")
        if png:
            encoded = "".join(png) if isinstance(png, list) else png
            destination.write_bytes(base64.b64decode(encoded))
            return
    raise ValueError(f"No PNG output found for cell {cell.get('id')} ({destination.name})")


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells_by_id = {cell["id"]: cell for cell in notebook["cells"] if "id" in cell}
    OUTPUT_DIR.mkdir(exist_ok=True)

    for cell_id, filename in FIGURES.items():
        cell = cells_by_id.get(cell_id)
        if cell is None:
            raise KeyError(
                f"Cell id {cell_id!r} (expected to produce {filename}) no longer exists in "
                f"the notebook -- find its new id and update FIGURES."
            )
        export_png(cell, OUTPUT_DIR / filename)
        print(f"Exported {filename} from cell {cell_id}")


if __name__ == "__main__":
    main()
