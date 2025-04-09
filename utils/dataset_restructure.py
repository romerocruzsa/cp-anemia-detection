import os
import pandas as pd
import shutil
import os
import pandas as pd
import shutil

def main():
    # === CONFIG ===
    DATA_DIR = os.path.expanduser("data/data")  # Path to the data directory
    CSV_PATH = os.path.join(DATA_DIR, "metadata.csv")  # Path to the CSV file
    IMAGES_DIR = os.path.join(DATA_DIR, "photo/")         # Directory with image files
    OUTPUT_DIR = os.path.expanduser("fingernail-anemia/")
    THRESHOLD = 120.0              # Hb level threshold in g/L

    # === LOAD DATA ===
    df = pd.read_csv(CSV_PATH)

    # === CREATE OUTPUT FOLDERS ===
    anemic_dir = os.path.join(OUTPUT_DIR, "Anemic")
    non_anemic_dir = os.path.join(OUTPUT_DIR, "Non-anemic")

    os.makedirs(anemic_dir, exist_ok=True)
    os.makedirs(non_anemic_dir, exist_ok=True)

    # === PROCESS EACH ROW ===
    for idx, row in df.iterrows():
        patient_id = str(row["PATIENT_ID"])
        hb_level = float(row["HB_LEVEL_GperL"])
        
        # Build expected image filename (assumes .jpg)
        image_name = f"{patient_id}.jpg"
        src_image_path = os.path.join(IMAGES_DIR, image_name)
        
        # Decide the category
        if hb_level < THRESHOLD:
            dest_dir = anemic_dir
        else:
            dest_dir = non_anemic_dir

        # Copy the image to the correct folder
        dest_image_path = os.path.join(dest_dir, image_name)
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dest_image_path)
            print(f"Moved {image_name} ➜ {dest_dir}")
        else:
            print(f"⚠️ Image not found: {src_image_path}")

    print("\n✅ Dataset reorganized successfully.")
    print(f"[✓] Reorganized dataset written to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()