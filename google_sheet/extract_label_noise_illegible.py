"""Extract dataset_name and image_id for 'label noise' or 'illegible' entries
from the 'error analysis_small' sheet in VLO_PARSeq.xlsx."""

import openpyxl
import csv
import os

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "VLO_PARSeq.xlsx")
SHEET_NAME = "error analysis_small"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "label_noise_illegible.csv")


def extract():
    wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True)
    ws = wb[SHEET_NAME]

    results = []
    for row in ws.iter_rows(min_row=2, max_col=6, values_only=True):
        dataset_name, image_id, _pred, _gt, label_noise, illegible = row
        if label_noise or illegible:
            img_id = int(image_id) if isinstance(image_id, float) else image_id
            reason = []
            if label_noise:
                reason.append("label_noise")
            if illegible:
                reason.append("illegible")
            results.append((dataset_name, img_id, ",".join(reason)))

    wb.close()

    # Save to CSV
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_name", "image_id", "reason"])
        writer.writerows(results)

    # Print results
    print(f"Total: {len(results)} entries")
    print(f"{'dataset_name':<20} {'image_id':<10} {'reason'}")
    print("-" * 50)
    for name, img_id, reason in results:
        print(f"{name:<20} {img_id:<10} {reason}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    extract()
