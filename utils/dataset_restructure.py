import os
import pandas as pd
import shutil

# Paths
CSV_PATH = os.path.expanduser("~/cp-anemia-detection/data/Eyes-Defy_Data_File_List.csv")
ITALY_XLSX = os.path.expanduser("~/cp-anemia-detection/data/eyes-defy-anemia-original/Italy/Italy.xlsx")
INDIA_XLSX = os.path.expanduser("~/cp-anemia-detection/data/eyes-defy-anemia-original/India/India.xlsx")
TARGET_DIR = os.path.expanduser("~/cp-anemia-detection/data/eyes-defy-anemia")
ANEMIC_THRESHOLD = 11.0

def setup_dirs():
    os.makedirs(os.path.join(TARGET_DIR, "Anemic/conjunctiva"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "Anemic/palpebral"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "Non-anemic/conjunctiva"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "Non-anemic/palpebral"), exist_ok=True)

def load_metadata():
    df_italy = pd.read_excel(ITALY_XLSX)
    df_italy["Source"] = "Italy"

    df_india = pd.read_excel(INDIA_XLSX)
    df_india["Source"] = "India"

    combined = pd.concat([df_italy, df_india], ignore_index=True)
    combined["Number"] = combined["Number"].astype(int)
    return combined

def restructure_dataset(file_df, meta_df):
    grouped = file_df.groupby(["Region", "Folder"])

    new_metadata = []

    for (region, folder), group in grouped:
        number = int(folder)
        prefix = f"{region.lower()}_{number:03d}"
        entry = meta_df[(meta_df["Source"] == region) & (meta_df["Number"] == number)]

        if entry.empty:
            print(f"[!] Missing metadata for {region} #{number}")
            continue

        row = entry.iloc[0]
        hb_raw = row.get("Hgb") or row.get("Hb")
        try:
            hb = float(hb_raw)
        except (TypeError, ValueError):
            hb = None

        label = "Anemic" if hb is not None and hb < ANEMIC_THRESHOLD else "Non-anemic"
        target_folder = os.path.join(TARGET_DIR, label)

        image_name = f"{prefix}.jpg"
        masks = {"palpebral": None, "forniceal": None, "forniceal_palpebral": None}

        for _, file_row in group.iterrows():
            src = file_row["FullPath"]
            filename = file_row["Filename"].lower()

            if filename.endswith(".jpg") and "palpebral" not in filename:
                dst = os.path.join(target_folder+"/conjunctiva", image_name)
                shutil.copy2(src, dst)

            elif filename.endswith("palpebral.png"):
                masks["palpebral"] = f"{prefix}_palpebral.png"
                shutil.copy2(src, os.path.join(target_folder+"/palpebral", masks["palpebral"]))

            # elif filename.endswith("forniceal.png"):
            #     masks["forniceal"] = f"{prefix}_forniceal.png"
            #     shutil.copy2(src, os.path.join(target_folder, masks["forniceal"]))

            # elif filename.endswith("forniceal_palpebral.png"):
            #     masks["forniceal_palpebral"] = f"{prefix}_forniceal_palpebral.png"
            #     shutil.copy2(src, os.path.join(target_folder, masks["forniceal_palpebral"]))

        new_metadata.append({
            "Number": number,
            "Source": region,
            "Hb": hb,
            "Age": row.get("Age", "NA"),
            "Sex": row.get("Sex", "NA"),
            "Label": label,
            "Image": image_name,
            # "Palpebral": masks["palpebral"],
            # "Forniceal": masks["forniceal"],
            # "Forniceal+Palpebral": masks["forniceal_palpebral"]
        })

    return new_metadata

def main():
    setup_dirs()
    file_df = pd.read_csv(CSV_PATH)
    meta_df = load_metadata()

    metadata = restructure_dataset(file_df, meta_df)

    df_out = pd.DataFrame(metadata)
    df_out.sort_values(by=["Source", "Number"], inplace=True)
    df_out.to_csv(os.path.join(TARGET_DIR, "metadata.csv"), index=False)

    print(f"[✓] Reorganized dataset written to: {TARGET_DIR}")

if __name__ == "__main__":
    main()

# import os
# import csv

# # Dataset root
# DATA_ROOT = os.path.expanduser("~/cp-anemia-detection/data/eyes-defy-anemia-original")
# OUTPUT_CSV = os.path.expanduser("~/cp-anemia-detection/data/eyes_defy_anemia_file_list.csv")

# def list_all_files():
#     rows = []

#     for region in ["Italy", "India"]:
#         region_path = os.path.join(DATA_ROOT, region)
#         print(f"\n=== {region.upper()} REGION ===")

#         if not os.path.isdir(region_path):
#             print(f"[!] Missing region folder: {region_path}")
#             continue

#         subfolders = sorted([
#             folder for folder in os.listdir(region_path)
#             if folder.isdigit() and os.path.isdir(os.path.join(region_path, folder))
#         ], key=lambda x: int(x))

#         for folder in subfolders:
#             folder_path = os.path.join(region_path, folder)
#             for file in sorted(os.listdir(folder_path)):
#                 file_path = os.path.join(folder_path, file)
#                 rows.append({
#                     "Region": region,
#                     "Folder": folder,
#                     "Filename": file,
#                     "FullPath": file_path
#                 })

#     # Write to CSV
#     os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
#     with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
#         fieldnames = ["Region", "Folder", "Filename", "FullPath"]
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)

#     print(f"\n[✓] File list saved to {OUTPUT_CSV}")

# if __name__ == "__main__":
#     list_all_files()

