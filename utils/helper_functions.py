"""
Helper functions.
"""
import subprocess
import numpy as np
from scipy.ndimage import label
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
import wandb
import os
from pathlib import Path
from collections import defaultdict
import json
import shutil

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    if rc != 0:
        print(f"Error executing command: {command}")
        print(process.stderr.read())
        return False
    return True

def prepare_nnunet_data(nnunet_raw, base_dir, task_name, max_cases=None):
    print(f"Preparing data for task: {task_name}")
    print(f"nnUNet raw data directory: {nnunet_raw}")
    print(f"Base directory: {base_dir}")
    print(f"Maximum cases to load: {'All' if max_cases is None else max_cases}")

    # Ensure the nnUNet directory structure exists
    for subdir in ["imagesTr", "labelsTr", "imagesTs"]:
        os.makedirs(nnunet_raw / subdir, exist_ok=True)
        print(f"Created directory: {nnunet_raw / subdir}")

    # Function to safely copy a file
    def safe_copy(src, dst):
        try:
            shutil.copyfile(src, dst)
        except PermissionError:
            print(f"Permission error when copying {src}. Skipping this file.")
        except Exception as e:
            print(f"Error copying {src}: {str(e)}")

    # Function to group files by case (including scan ID)
    def group_files_by_case(files):
        cases = defaultdict(list)
        for file in files:
            case_id = file.name.split('-')[0] + '_' + file.name.split('-')[1]
            cases[case_id].append(file)
        return cases

    # Define modality order and mapping
    modality_order = ['t1n', 't1c', 't2w', 't2f']
    modality_mapping = {'t1n': '0000', 't1c': '0001', 't2w': '0002', 't2f': '0003'}

    # Copy and rename training and validation images and labels
    training_cases = []
    case_count = 0
    for subset in ["training", "validation"]:
        print(f"Processing {subset} data...")
        image_files = list((base_dir / "images" / subset).glob("*.nii.gz"))
        print(f"Found {len(image_files)} image files in {subset}")
        grouped_images = group_files_by_case(image_files)
        
        for case_id, images in grouped_images.items():
            if max_cases is not None and case_count >= max_cases:
                print(f"Reached maximum number of cases ({max_cases}). Stopping data loading.")
                break
            
            print(f"Processing case: {case_id}")
            # Sort images by modality order
            images.sort(key=lambda x: modality_order.index(x.name.split('-')[-1].split('.')[0]))
            
            # Copy and rename images
            for img in images:
                modality = img.name.split('-')[-1].split('.')[0]
                new_name = f"{case_id}_{modality_mapping[modality]}.nii.gz"
                safe_copy(img, nnunet_raw / "imagesTr" / new_name)
            
            # Find and copy label file
            label_file = None
            label_patterns = [f"{case_id.replace('_', '-')}-seg.nii.gz", f"{case_id.replace('_', '-')}*seg.nii.gz", f"{case_id.replace('_', '-')}*.nii.gz"]
            for pattern in label_patterns:
                label_files = list((base_dir / "labels" / subset).glob(pattern))
                if label_files:
                    label_file = label_files[0]
                    break
            
            if label_file:
                new_label_name = f"{case_id}.nii.gz"
                safe_copy(label_file, nnunet_raw / "labelsTr" / new_label_name)
                training_cases.append(case_id)
                case_count += 1
                print(f"Processed case {case_count}/{max_cases if max_cases is not None else 'All'}")
            else:
                print(f"Warning: No label file found for case {case_id} in {subset} subset.")

            if max_cases is not None and case_count >= max_cases:
                print(f"Reached maximum number of cases ({max_cases}). Stopping data loading.")
                break

        if max_cases is not None and case_count >= max_cases:
            break

    # Copy and rename test images
    test_cases = []
    print("Processing test data...")
    test_images = list((base_dir / "images" / "testing").glob("*.nii.gz"))
    print(f"Found {len(test_images)} test image files")
    grouped_test_images = group_files_by_case(test_images)
    
    test_case_count = 0
    for case_id, images in grouped_test_images.items():
        if max_cases is not None and test_case_count >= max_cases:
            print(f"Reached maximum number of test cases ({max_cases}). Stopping test data loading.")
            break
        
        print(f"Processing test case: {case_id}")
        images.sort(key=lambda x: modality_order.index(x.name.split('-')[-1].split('.')[0]))
        for img in images:
            modality = img.name.split('-')[-1].split('.')[0]
            new_name = f"{case_id}_{modality_mapping[modality]}.nii.gz"
            safe_copy(img, nnunet_raw / "imagesTs" / new_name)
        test_cases.append(case_id)
        test_case_count += 1
        print(f"Processed test case {test_case_count}/{max_cases if max_cases is not None else 'All'}")

    # Create dataset.json
    dataset_json = {
        "name": task_name,
        "description": "BraTS 2024 Post-Surgery Segmentation",
        "tensorImageSize": "4D",
        "reference": "https://www.synapse.org/#!Synapse:syn51156910/wiki/621615",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0 04/11/2023",
        "modality": {
            "0": "T1",
            "1": "T1ce",
            "2": "T2",
            "3": "FLAIR"
        },
        "labels": {
            "0": "background",
            "1": "non-enhancing tumor core (NETC)",
            "2": "surrounding non-enhancing FLAIR hyperintensity (SNFH)",
            "3": "enhancing tissue (ET)",
            "4": "resection cavity (RC)"
        },
        "numTraining": len(training_cases),
        "numTest": len(test_cases),
        "training": [{"image": f"./imagesTr/{case_id}.nii.gz", "label": f"./labelsTr/{case_id}.nii.gz"} for case_id in training_cases],
        "test": [f"./imagesTs/{case_id}.nii.gz" for case_id in test_cases]
    }

    dataset_json_path = nnunet_raw / "dataset.json"
    with open(dataset_json_path, "w", encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=4)
    print(f"Created dataset.json at {dataset_json_path}")

    print(f"Data preparation completed for {task_name}.")
    print(f"Task folder updated at: {nnunet_raw}")
    print(f"Total training cases: {len(training_cases)}")
    print(f"Total test cases: {len(test_cases)}")

    # Verify the contents of the task folder
    print("\nVerifying task folder contents:")
    for subdir in ["imagesTr", "labelsTr", "imagesTs"]:
        subdir_path = nnunet_raw / subdir
        file_count = len(list(subdir_path.glob("*.nii.gz")))
        print(f"{subdir}: {file_count} files")
    
    if (nnunet_raw / "dataset.json").exists():
        print("dataset.json: Present")
    else:
        print("dataset.json: Missing")

    print("\nPlease check the above information to ensure all data has been correctly prepared.")
    
def calculate_metrics(pred, gt):
    def dice_coefficient(y_true, y_pred):
        intersection = np.sum(y_true * y_pred)
        return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

    def hausdorff_distance_95(y_true, y_pred):
        d1 = directed_hausdorff(y_true, y_pred)[0]
        d2 = directed_hausdorff(y_pred, y_true)[0]
        return max(np.percentile(d1, 95), np.percentile(d2, 95))

    lesion_pred, _ = label(pred)
    lesion_gt, _ = label(gt)

    unique_lesions_pred = np.unique(lesion_pred)[1:]  # Exclude background
    unique_lesions_gt = np.unique(lesion_gt)[1:]  # Exclude background

    dsc_scores = []
    hd95_scores = []

    for lesion_id in unique_lesions_gt:
        gt_lesion = (lesion_gt == lesion_id).astype(int)
        pred_lesion = (lesion_pred == lesion_id).astype(int)
        
        dsc = dice_coefficient(gt_lesion, pred_lesion)
        hd95 = hausdorff_distance_95(gt_lesion, pred_lesion)
        
        dsc_scores.append(dsc)
        hd95_scores.append(hd95)

    return np.mean(dsc_scores), np.mean(hd95_scores)

def evaluate_fold(fold_output_dir, base_dir):
    fold_output_dir = Path(fold_output_dir)
    base_dir = Path(base_dir)
    
    pred_dir = fold_output_dir / "validation_raw"
    gt_dir = base_dir / "labels" / "validation"  # Adjusted path based on our earlier structure
    
    dsc_scores = []
    hd95_scores = []
    
    for pred_file in pred_dir.glob("*.nii.gz"):
        gt_file = gt_dir / pred_file.name
        
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_file)))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_file)))
        
        dsc, hd95 = calculate_metrics(pred, gt)
        dsc_scores.append(dsc)
        hd95_scores.append(hd95)
    
    return np.mean(dsc_scores), np.mean(hd95_scores)

def log_metrics_to_wandb(metrics, step=None):
    for key, value in metrics.items():
        wandb.log({key: value}, step=step)
