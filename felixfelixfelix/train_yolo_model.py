# -*- coding: utf-8 -*-
"""
train_yolo_model.py

Purpose: A consolidated script to tune, train, validate, and test a YOLOv8 model.
This version is configured for the yolov8n-p2 architecture for best performance.
"""

# pip install torch ultralytics onnx onnxruntime matplotlib opencv-python Pillow pyyaml --quiet

import time
import torch
import numpy as np
from ultralytics import YOLO
import os
import yaml
import logging
from pathlib import Path
import ultralytics.data.build as build
from ultralytics.data.build import YOLODataset

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Critical User Configuration ---
YOLO_CLASS_NAMES = [
    "cargo_aircraft", "commercial_aircraft", "drone", "fighter_jet", "fighter_plane",
    "helicopter", "light aircraft", "missile", "truck", "car", "tank", "bus",
    "van", "cargo_ship", "yacht", "cruise_ship", "warship", "sailboat"
]
YOLO_DATASET_ROOT_PATH = "./yolo_dataset_output"
YOLO_DATASET_CONFIG_FILENAME = "yolo_18_classes_config.yaml"

# --- Weighted Dataloader for Class Imbalance ---
class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix
        if self.train_mode:
            self.count_instances()
            class_weights = np.sum(self.counts) / (self.counts + 1)
            self.agg_func = np.mean
            self.class_weights = np.array(class_weights)
            self.weights = self.calculate_weights()
            self.probabilities = self.calculate_probabilities()
    def count_instances(self):
        self.counts = np.zeros(len(self.data["names"]), dtype=int)
        for label in self.labels:
            cls = label['cls'].flatten().astype(int)
            for cid in cls: self.counts[cid] += 1
    def calculate_weights(self):
        weights = []
        for label in self.labels:
            cls = label['cls'].flatten().astype(int)
            if cls.size == 0: weights.append(1); continue
            weight = self.agg_func(self.class_weights[cls]); weights.append(weight)
        return weights
    def calculate_probabilities(self):
        total_weight = sum(self.weights)
        if total_weight == 0: return np.ones(len(self.weights)) / len(self.weights)
        return [w / total_weight for w in self.weights]
    def __getitem__(self, index):
        if not self.train_mode: return self.transforms(self.get_image_and_label(index))
        weighted_index = np.random.choice(len(self.labels), p=self.probabilities)
        return self.transforms(self.get_image_and_label(weighted_index))

# To enable the weighted dataloader, uncomment the line below
# build.YOLODataset = YOLOWeightedDataset
# logging.info("Patched Ultralytics YOLODataset with custom YOLOWeightedDataset.")

# --- Utility Functions ---
def get_device_string():
    if torch.cuda.is_available(): return 'cuda'
    return 'cpu'

def create_yolo_dataset_config_file(dataset_root_path, class_names_list, config_file_name):
    abs_dataset_root_path = os.path.abspath(dataset_root_path)
    config_file_abs_path = os.path.abspath(config_file_name)
    data = {'path': abs_dataset_root_path, 'train': 'train/images', 'val': 'val/images', 'nc': len(class_names_list), 'names': class_names_list}
    with open(config_file_abs_path, 'w') as f: yaml.dump(data, f, sort_keys=False, default_flow_style=None)
    logging.info(f"YOLO dataset config file created at: {config_file_abs_path}")
    return config_file_abs_path

# --- Core Pipeline Functions ---

def tune_yolo_hyperparameters(dataset_config_path, model_variant, image_size, device_str):
    try:
        model = YOLO(model_variant)
        logging.info(f"--- Starting YOLOv8 Hyperparameter Tuning ---")
        model.tune(
            data=dataset_config_path,
            epochs=10,      # Epochs per trial
            iterations=200,   # Number of trials. More is better.
            optimizer='auto',
            project='YOLOv8_P2_Tuning',
            name='tune_run_final',
            imgsz=image_size,
            batch=-1,       # Auto-batch to maximize VRAM usage
            device=device_str,
            plots=True
        )
        logging.info(f"--- Hyperparameter tuning complete! ---")
    except Exception as e:
        logging.error(f"Error during YOLO hyperparameter tuning: {e}", exc_info=True)

def train_yolo(dataset_config_path, model_variant, epochs, batch_size, image_size, device_str, export_interval_epochs, **kwargs):
    try:
        yolo_model = YOLO(model_variant)
        logging.info(f"--- Starting YOLOv8 Training ---")
        
        epoch_durations = []
        def on_epoch_end_callback(trainer):
            # Time Estimation Logic
            if hasattr(trainer, 'epoch_time') and trainer.epoch_time is not None:
                epoch_durations.append(trainer.epoch_time)
            if epoch_durations:
                avg_epoch_time = sum(epoch_durations) / len(epoch_durations)
                current_epoch = trainer.epoch + 1
                epochs_remaining = trainer.epochs - current_epoch
                if epochs_remaining > 0:
                    eta_seconds = epochs_remaining * avg_epoch_time
                    eta_str = time.strftime("%Hh:%Mm:%Ss", time.gmtime(eta_seconds))
                    logging.info(f"Epoch {current_epoch}/{trainer.epochs} complete. Est. time remaining: {eta_str}")

            # Periodic Saving Logic
            current_epoch = trainer.epoch + 1
            if export_interval_epochs > 0 and current_epoch % export_interval_epochs == 0:
                backup_path = Path(trainer.save_dir) / 'weights' / f'epoch_{current_epoch}.pt'
                try:
                    # Use the main yolo_model object to save
                    yolo_model.save(str(backup_path))
                    logging.info(f"Callback: Saved backup model to {backup_path}")
                except Exception as e:
                    logging.error(f"Callback ERROR: Could not save backup model at epoch {current_epoch}: {e}")

        yolo_model.add_callback("on_fit_epoch_end", on_epoch_end_callback)

        results = yolo_model.train(
            data=dataset_config_path, epochs=epochs, batch=batch_size, imgsz=image_size,
            device=device_str,
            project='My_YOLOv8_18Class_P2_Project',
            name='train_run_final',
            **kwargs
        )
        
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        logging.info(f"YOLOv8 training complete. Best model saved at: {best_model_path}")
        return str(best_model_path)
    except Exception as e:
        logging.error(f"Error during YOLO model training: {e}", exc_info=True)
        return None

def validate_yolo(model_path, dataset_config_path, image_size, device_str):
    try:
        model = YOLO(model_path)
        logging.info(f"--- Validating YOLOv8 model: {model_path} ---")
        metrics = model.val(data=dataset_config_path, imgsz=image_size, device=device_str, split='val')
        logging.info(f"Validation mAP50-95: {metrics.box.map}")
        return metrics
    except Exception as e:
        logging.error(f"Error during YOLO model validation: {e}", exc_info=True)
        return None

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    MODEL_VARIANT = 'yolov8n-p2.yaml'
    IMAGE_SIZE = 640
    BATCH_SIZE = 8      # Adjust based on your GPU's VRAM
    EPOCHS = 300      # Number of epochs for the final training run
    EXPORT_INTERVAL = 20 # Save a backup every 20 epochs
    
    # --- Control Flags ---
    SHOULD_TUNE = False
    SHOULD_TRAIN = True
    SHOULD_VALIDATE = False

    # --- Setup ---
    current_device = get_device_string()
    logging.info(f"--- Using device: {current_device} ---")
    if not os.path.isdir(YOLO_DATASET_ROOT_PATH):
        logging.error(f"Dataset path not found: {YOLO_DATASET_ROOT_PATH}"); exit()
    dataset_yaml_path = create_yolo_dataset_config_file(YOLO_DATASET_ROOT_PATH, YOLO_CLASS_NAMES, YOLO_DATASET_CONFIG_FILENAME)

    # --- Pipeline ---
    if SHOULD_TUNE:
        tune_yolo_hyperparameters(dataset_yaml_path, MODEL_VARIANT, IMAGE_SIZE, current_device)
        logging.info("Tuning finished. Please copy new HPs into the 'hyperparams' dictionary, set SHOULD_TUNE=False, and run again.")
        exit()

    trained_model_path = None
    if SHOULD_TRAIN:
        # PASTE YOUR TUNED HYPERPARAMETERS HERE after running the tuning step
        hyperparams = {
            'patience': 50, 'optimizer': 'Adam', 'lr0': 0.01092, 'lrf': 0.01296, 'momentum': 0.98,
            'weight_decay': 0.00035, 'warmup_epochs': 3.17567, 'warmup_momentum': 0.69645,
            'box': 5.4976, 'cls': 0.69935, 'dfl': 1.26811, 'hsv_h': 0.0124, 'hsv_s': 0.70534,
            'hsv_v': 0.49175, 'degrees': 0.0, 'translate': 0.08286, 'scale': 0.44177,
            'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.45064,
            'mosaic': 1.0, 'mixup': 0.0, 'close_mosaic': 10
        }
        trained_model_path = train_yolo(
            dataset_yaml_path, MODEL_VARIANT, EPOCHS, BATCH_SIZE, IMAGE_SIZE, 
            current_device, EXPORT_INTERVAL, **hyperparams
        )
        if not trained_model_path:
            logging.error("Training failed."); exit()
    
    if SHOULD_VALIDATE:
        # If you skipped training, specify the path to your best model here
        if not trained_model_path:
            trained_model_path = "My_YOLOv8_18Class_P2_Project/train_run_final/weights/best.pt"
        
        if os.path.exists(trained_model_path):
            validate_yolo(trained_model_path, dataset_yaml_path, IMAGE_SIZE, current_device)
        else:
            logging.error(f"Cannot validate. Model file not found at: {trained_model_path}")

    logging.info("\n--- Main script execution finished. ---")
    
