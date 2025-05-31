import os
import zipfile
import shutil
from PIL import Image
import pillow_heif
import csv
import numpy as np
import cv2  # Importing OpenCV module for image processing
from tqdm.notebook import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch

# Register HEIC handler for Pillow
pillow_heif.register_heif_opener()

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_zip_files(zips_path):
    zip_files = [item for item in os.listdir(zips_path) if item.lower().endswith('.zip')]
    total_files = len(zip_files)
    
    if total_files == 0:
        print("No ZIP files found for extraction.")
        return
    
    print(f"Found {total_files} ZIP files to extract.")
    
    for item in tqdm(zip_files, desc="Extracting ZIP files", unit="file"):
        zip_path = os.path.join(zips_path, item)
        extract_dir = os.path.join(zips_path, os.path.splitext(item)[0].strip())
        
        if os.path.exists(extract_dir):
            continue  # Skip extraction if folder exists
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted {zip_path} to {extract_dir}")
        except FileNotFoundError as e:
            print(f"Failed to extract {zip_path}: {e}")
        except Exception as e:
            print(f"An error occurred while extracting {zip_path}: {e}")

def process_images_in_folders(input_path):
    for root, dirs, files in os.walk(input_path):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            if any(f.lower().endswith(('.heic', '.mp4', '.jpg', '.jpeg', '.mov')) for f in os.listdir(folder_path)):
                print(f"Processing images in {folder_path}")
                process_images(folder_path)

def convert_heic_to_jpg(image_folder):
    heic_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.heic')]
    if heic_files:
        for image_file in tqdm(heic_files, desc="Converting HEIC to JPG"):
            heic_path = os.path.join(image_folder, image_file)
            jpg_path = os.path.join(image_folder, os.path.splitext(image_file)[0] + '.jpg')
            
            if os.path.exists(jpg_path):
                continue
            
            image = Image.open(heic_path)
            image.save(jpg_path, "JPEG", quality=100)
            os.remove(heic_path)

def delete_movie_files(image_folder):
    movie_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.mp4', '.mov'))]
    for movie_file in movie_files:
        os.remove(os.path.join(image_folder, movie_file))

def evaluate_clip_concept(image_path, concept):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(image_path)
    inputs = clip_processor(text=[concept], images=image, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    clip_model.to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
    
    return logits_per_image[0][0].item()

def extract_features(image_path):
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.numpy().flatten()

def compute_exposure(image_np):
    hist, _ = np.histogram(image_np, bins=256, range=(0, 256))
    overexposure = np.sum(hist[-10:])
    underexposure = np.sum(hist[:10])
    return overexposure + underexposure

def load_existing_csv(output_csv):
    existing_data = {}
    if os.path.exists(output_csv):
        with open(output_csv, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)
            for row in csvreader:
                filename = row[0]
                metrics = [float(value) for value in row[1:]]
                existing_data[filename] = metrics
    return existing_data

def compute_metrics(image_path, all_features=None):
    image = Image.open(image_path).convert('L')
    image_np = np.array(image)
    
    laplacian_sharpness = cv2.Laplacian(image_np, cv2.CV_64F).var()
    exposure = compute_exposure(image_np)
    clip_blurryness = evaluate_clip_concept(image_path, "a blurry photo")
    clip_exposure = evaluate_clip_concept(image_path, "a poorly lit photo")

    if all_features is not None:
        current_features = extract_features(image_path)
        avg_features = np.mean(all_features, axis=0)
        clip_outlier_detection = np.linalg.norm(current_features - avg_features)
    else:
        clip_outlier_detection = 0
    
    return laplacian_sharpness, exposure, clip_blurryness, clip_exposure, clip_outlier_detection

def normalize_by_sum(data):
    column_sums = np.sum(data, axis=0)
    normalized_data = data / column_sums
    return normalized_data

def process_images(image_folder, convert_heic=True, delete_movies=True):
    output_csv = os.path.join(image_folder, "output_sharpness.csv")
    
    existing_data = load_existing_csv(output_csv)
    
    if convert_heic:
        convert_heic_to_jpg(image_folder)
    
    if delete_movies:
        delete_movie_files(image_folder)
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    sharpness_data = []
    new_images = [image_file for image_file in image_files if image_file not in existing_data]

    if new_images:
        all_features = [extract_features(os.path.join(image_folder, image_file)) for image_file in tqdm(image_files, desc="Extracting features for outlier detection")]

        with open(output_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['filename', 'laplacian_sharpness', 'exposure', 'clip_blurryness', 'clip_exposure', 'clip_outlier_detection'])

            for image_file in tqdm(image_files, desc="Evaluating quality metrics"):
                if image_file in existing_data:
                    sharpness_data.append([image_file] + existing_data[image_file])
                else:
                    image_path = os.path.join(image_folder, image_file)
                    metrics = compute_metrics(image_path, all_features)
                    sharpness_data.append([image_file, *metrics])

            sharpness_array = np.array([row[1:] for row in sharpness_data])
            normalized_array = normalize_by_sum(sharpness_array)

            for i, row in enumerate(sharpness_data):
                sharpness_data[i] = [row[0]] + list(normalized_array[i])
                csvwriter.writerow(sharpness_data[i])
    else:
        print("No new images detected. No re-computation necessary.")

def main(zip_path=None, folder_path=None):
    if zip_path and folder_path:
        choice = input("Both ZIP and folder paths provided. Which one do you want to process? (zip/folder): ").strip().lower()
        if choice == "zip":
            extract_zip_files(zip_path)
            print(f"Extracted ZIP files from {zip_path}")
            process_images_in_folders(zip_path)
        elif choice == "folder":
            print(f"Processing images in folder: {folder_path}")
            process_images_in_folders(folder_path)
        else:
            print("Invalid input. Please choose 'zip' or 'folder'.")
    elif zip_path:
        extract_zip_files(zip_path)
        print(f"Extracted ZIP files from {zip_path}")
        process_images_in_folders(zip_path)
    elif folder_path:
        print(f"Processing images in folder: {folder_path}")
        process_images_in_folders(folder_path)
    else:
        print("Please provide at least one path (ZIP or folder).")

# Example usage
if __name__ == "__main__":
    zips_path = r"C:\Users\alons\Desktop\MesoAmerica_Photogrammetry\PLAYGROUND"
    folder_path = r"C:\path\to\your\image\folder"  # Optional
    main(zip_path=zips_path, folder_path=folder_path)
