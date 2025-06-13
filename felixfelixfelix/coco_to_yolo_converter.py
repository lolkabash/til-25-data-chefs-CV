import os
import json
from pathlib import Path
import shutil
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TARGET_CLASS_NAMES = [
    "cargo_aircraft", "commercial_aircraft", "drone", "fighter_jet", "fighter_plane",
    "helicopter", "light aircraft", "missile", "truck", "car", "tank", "bus",
    "van", "cargo_ship", "yacht", "cruise_ship", "warship", "sailboat"
]
# Corrected the class count check
if len(TARGET_CLASS_NAMES) != 18:
    logging.warning(f"WARNING: Expected 18 classes, but found {len(TARGET_CLASS_NAMES)}.")

COCO_JSON_PATH = "../advanced/cv/annotation.json"
ALL_IMAGES_DIR = "../advanced/cv/images"
YOLO_DATASET_OUTPUT_DIR = "./yolo_dataset_output"
TRAIN_RATIO = 0.8

"""Convert COCO to YOLO Bounding Box"""

def coco_to_yolo_bbox(bbox, img_width, img_height):
    x_min, y_min, w, h = bbox
    if img_width == 0 or img_height == 0: return None
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    return x_center, y_center, width_norm, height_norm

"""Convert COCO to YOLO"""

def convert_coco_to_yolo():
    logging.info("Starting COCO to YOLO conversion...")
    logging.info(f"  Target Classes: {TARGET_CLASS_NAMES}")
    logging.info(f"  COCO JSON Path: {COCO_JSON_PATH}")
    logging.info(f"  Source Images Dir: {ALL_IMAGES_DIR}")
    logging.info(f"  Output YOLO Dataset Dir: {YOLO_DATASET_OUTPUT_DIR}")

    if not os.path.exists(COCO_JSON_PATH):
        logging.error(f"COCO JSON file not found at: {COCO_JSON_PATH}")
        return
    if not os.path.isdir(ALL_IMAGES_DIR):
        logging.error(f"Source images directory not found at: {ALL_IMAGES_DIR}")
        return

    # Create output directories
    for split in ["train", "val"]:
        Path(os.path.join(YOLO_DATASET_OUTPUT_DIR, split, "images")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(YOLO_DATASET_OUTPUT_DIR, split, "labels")).mkdir(parents=True, exist_ok=True)

    with open(COCO_JSON_PATH, 'r') as f:
        coco_data = json.load(f)

    # --- Category Mapping ---
    # 1. Map COCO category IDs to their names from the JSON
    coco_cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    # 2. Map your target class names to 0-indexed YOLO class IDs
    target_name_to_yolo_id = {name: i for i, name in enumerate(TARGET_CLASS_NAMES)}

    # 3. Create the final mapping from COCO category ID -> YOLO class ID
    #    Only include classes that are in your TARGET_CLASS_NAMES
    coco_id_to_yolo_id = {}
    found_coco_categories = []
    for coco_id, coco_name in coco_cat_id_to_name.items():
        if coco_name in target_name_to_yolo_id:
            coco_id_to_yolo_id[coco_id] = target_name_to_yolo_id[coco_name]
            if coco_name not in found_coco_categories:
                 found_coco_categories.append(coco_name)
        else:
            logging.debug(f"COCO category '{coco_name}' (ID: {coco_id}) is not in TARGET_CLASS_NAMES and will be skipped.")

    # Check if all target classes were found in the COCO JSON
    missing_target_classes = [name for name in TARGET_CLASS_NAMES if name not in found_coco_categories]
    if missing_target_classes:
        logging.warning(f"The following TARGET_CLASS_NAMES were NOT found in the COCO JSON categories: {missing_target_classes}")
        logging.warning(f"Found COCO categories that were mapped: {found_coco_categories}")
        logging.warning("Annotations for missing classes will not be created. Please check your TARGET_CLASS_NAMES and COCO JSON.")


    # Group annotations by image_id
    annotations_by_image_id = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image_id:
            annotations_by_image_id[img_id] = []
        annotations_by_image_id[img_id].append(ann)

    images_info = coco_data.get('images', [])
    random.shuffle(images_info) # Shuffle for random train/val split
    split_index = int(len(images_info) * TRAIN_RATIO)

    processed_image_count = 0
    for i, img_info in enumerate(images_info):
        img_id = img_info['id']
        img_filename = img_info['file_name'] # Assumes file_name is just the name, not a path
        img_width = img_info.get('width')
        img_height = img_info.get('height')

        if img_width is None or img_height is None or img_width == 0 or img_height == 0:
            logging.warning(f"Image '{img_filename}' (ID: {img_id}) is missing width/height or has zero dimensions. Skipping.")
            # Optionally, try to open the image to get dimensions if not in JSON,
            # but this adds complexity and slows down conversion.
            # For now, we rely on COCO JSON having correct w/h.
            continue

        yolo_labels_for_this_image = []
        if img_id in annotations_by_image_id:
            for ann in annotations_by_image_id[img_id]:
                coco_category_id = ann['category_id']
                if coco_category_id in coco_id_to_yolo_id: # If this COCO category is one of our targets
                    yolo_class_id = coco_id_to_yolo_id[coco_category_id]
                    bbox_coco = ann['bbox'] # COCO [x_min, y_min, width, height]

                    yolo_bbox_coords = coco_to_yolo_bbox(bbox_coco, img_width, img_height)
                    if yolo_bbox_coords:
                        x_center, y_center, norm_width, norm_height = yolo_bbox_coords
                        yolo_labels_for_this_image.append(
                            f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                        )

        # Determine destination split folder
        current_split_folder = "train" if i < split_index else "val"

        # Define paths
        # Ensure img_filename does not contain subdirectories if ALL_IMAGES_DIR is flat
        source_image_full_path = os.path.join(ALL_IMAGES_DIR, os.path.basename(img_filename))

        output_label_filename = Path(img_filename).stem + ".txt"
        output_label_full_path = os.path.join(YOLO_DATASET_OUTPUT_DIR, current_split_folder, "labels", output_label_filename)

        output_image_filename = os.path.basename(img_filename) # Ensure only filename part
        output_image_full_path = os.path.join(YOLO_DATASET_OUTPUT_DIR, current_split_folder, "images", output_image_filename)

        # Write YOLO label file (only if there are relevant labels for this image)
        if yolo_labels_for_this_image:
            with open(output_label_full_path, 'w') as f_out:
                f_out.write("\n".join(yolo_labels_for_this_image))

            # Copy image only if its label file was created (meaning it has relevant annotations)
            if os.path.exists(source_image_full_path):
                shutil.copy2(source_image_full_path, output_image_full_path)
                processed_image_count +=1
            else:
                logging.warning(f"Source image file not found and not copied: {source_image_full_path}")
        # else:
            # logging.debug(f"No target annotations for image '{img_filename}' (ID: {img_id}). Image not copied, label file not created.")


    logging.info(f"Conversion complete. {processed_image_count} images with relevant annotations processed and copied.")
    logging.info(f"YOLO formatted dataset saved to: {os.path.abspath(YOLO_DATASET_OUTPUT_DIR)}")
    logging.info("Please verify the contents of the output directory, especially the 'train' and 'val' splits.")
    logging.info(f"Ensure your YOLO_DATASET_ROOT_PATH in train_yolo_model.py points to '{os.path.abspath(YOLO_DATASET_OUTPUT_DIR)}'")

if __name__ == '__main__':
    # Before running, ensure TARGET_CLASS_NAMES, COCO_JSON_PATH, ALL_IMAGES_DIR,
    # and YOLO_DATASET_OUTPUT_DIR are correctly set at the top of this script.
    convert_coco_to_yolo()