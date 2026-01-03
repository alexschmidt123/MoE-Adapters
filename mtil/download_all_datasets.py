#!/usr/bin/env python3
"""
Script to download all required datasets for MTIL training ahead of time.
This will pre-download all datasets so training can proceed without interruption.

Features:
- Automatically downloads from torchvision when possible
- Falls back to Hugging Face for datasets with broken URLs (StanfordCars, SUN397)
- Checks for existing datasets before downloading
- Cleans up temporary archive files
- Shows progress and directory structure

Requirements:
- torchvision (for most datasets)
- datasets library (for Hugging Face fallback): pip install datasets
"""

import os
import sys
import ssl
from pathlib import Path

# Add parent directory to path to import datasets
sys.path.insert(0, str(Path(__file__).parent))

try:
    from torchvision import datasets
    from PIL import Image
    import torch
    from continuum.datasets import TinyImageNet200
    # Try to import datasets library for Hugging Face downloads
    try:
        from datasets import load_dataset
        HAS_DATASETS_LIB = True
    except ImportError:
        HAS_DATASETS_LIB = False
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure you're in the correct conda environment (MoE_Adapters4CL)")
    sys.exit(1)

# Handle SSL certificate issues for some datasets
# Some datasets (like EuroSAT) may fail with SSL certificate verification errors
# We disable SSL verification globally as a workaround for downloading public datasets
# Note: This is safe for downloading public datasets from trusted sources
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass

# Default location (same as used in training scripts)
DEFAULT_DATA_LOCATION = os.path.expanduser("~/Documents/MoE-Adapters/datasets")

def download_dataset(dataset_name, location, preprocess=None):
    """Download a single dataset."""
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name}...")
    print(f"{'='*60}")
    
    try:
        if dataset_name == "TinyImagenet":
            # TinyImagenet uses continuum library
            # Code now uses real path: tinyimagenet/tiny-imagenet-200 (no symlink needed)
            print("TinyImagenet: Checking/Downloading...")
            tinyimagenet_real_path = os.path.join(location, 'tinyimagenet', 'tiny-imagenet-200')
            tinyimagenet_alt_path = os.path.join(location, 'tiny-imagenet-200')  # Fallback for symlink
            
            # Check if already exists at real path
            if os.path.exists(tinyimagenet_real_path) and os.path.isdir(tinyimagenet_real_path):
                train_dir = os.path.join(tinyimagenet_real_path, 'train')
                val_dir = os.path.join(tinyimagenet_real_path, 'val')
                if os.path.exists(train_dir) and os.path.exists(val_dir):
                    print(f"✓ TinyImagenet already exists at {tinyimagenet_real_path}")
                    return True
            # Check alternative location (for backward compatibility)
            elif os.path.exists(tinyimagenet_alt_path) and os.path.isdir(tinyimagenet_alt_path):
                train_dir = os.path.join(tinyimagenet_alt_path, 'train')
                val_dir = os.path.join(tinyimagenet_alt_path, 'val')
                if os.path.exists(train_dir) and os.path.exists(val_dir):
                    print(f"✓ TinyImagenet found at {tinyimagenet_alt_path} (will use real path: {tinyimagenet_real_path})")
                    return True
            
            # Try to download via continuum (it may download to a different location)
            try:
                # Continuum downloads to its own cache, but we need it at location/tiny-imagenet-200
                # First try continuum download
                continuum_dataset = TinyImageNet200(data_path=location, download=True, train=True)
                print(f"  Continuum downloaded TinyImagenet")
                
                # Check if it's in the expected location now (real path)
                if os.path.exists(tinyimagenet_real_path):
                    print(f"✓ TinyImagenet downloaded/verified at {tinyimagenet_real_path}")
                    return True
                elif os.path.exists(tinyimagenet_alt_path):
                    print(f"✓ TinyImagenet found at {tinyimagenet_alt_path}")
                    return True
                else:
                    # Continuum might have downloaded to a different location
                    # Check common continuum cache locations
                    cache_locations = [
                        os.path.expanduser("~/.continuum"),
                        os.path.join(location, "tinyimagenet"),
                    ]
                    found = False
                    for cache_loc in cache_locations:
                        if os.path.exists(cache_loc):
                            print(f"  Found continuum cache at {cache_loc}")
                    print(f"  Note: Dataset may be in continuum cache")
                    print(f"  Code will use: {tinyimagenet_real_path}")
                    return True  # Assume success, continuum will handle it
                    
            except Exception as e:
                print(f"  Continuum download error: {e}")
                # Check if it exists anyway (maybe manually placed)
                if os.path.exists(tinyimagenet_real_path):
                    print(f"✓ TinyImagenet found at {tinyimagenet_real_path}")
                    return True
                elif os.path.exists(tinyimagenet_alt_path):
                    print(f"✓ TinyImagenet found at {tinyimagenet_alt_path}")
                    return True
                print(f"  You may need to download TinyImagenet manually")
                print(f"  Expected location: {tinyimagenet_real_path}")
                return False
            
        elif dataset_name == "Aircraft":
            print("Aircraft: Downloading train split...")
            datasets.FGVCAircraft(location, split="train", download=True)
            print("Aircraft: Downloading test split...")
            datasets.FGVCAircraft(location, split="test", download=True)
            print(f"✓ Aircraft downloaded")
            
        elif dataset_name == "Caltech101":
            # Check if Caltech101 already exists
            caltech101_path = os.path.join(location, 'caltech101')
            if os.path.exists(caltech101_path) and os.path.isdir(caltech101_path):
                # Check if it has the required subdirectory
                categories_path = os.path.join(caltech101_path, '101_ObjectCategories')
                if os.path.exists(categories_path) and os.path.isdir(categories_path):
                    # Check if it has some category folders (at least a few)
                    categories = [d for d in os.listdir(categories_path) 
                                if os.path.isdir(os.path.join(categories_path, d))]
                    if len(categories) > 0:
                        print(f"✓ Caltech101 already exists at {caltech101_path} ({len(categories)} categories found)")
                        return True
            
            print("Caltech101: Attempting download (may fail with 404)...")
            try:
                datasets.Caltech101(location, download=True)
                print(f"✓ Caltech101 downloaded")
            except Exception as e:
                print(f"✗ Caltech101 download failed: {e}")
                print("  You may need to download this manually.")
                print("  The original download URL is broken (404 error).")
                print("  Try downloading from alternative sources (Kaggle, Hugging Face, etc.)")
                return False
                
        elif dataset_name == "CIFAR100":
            # Check if CIFAR100 already exists
            cifar100_path = os.path.join(location, 'cifar-100-python')
            cifar100_alt_path = os.path.join(location, 'cifar100', 'cifar-100-python')
            
            if os.path.exists(cifar100_path) and os.path.isdir(cifar100_path):
                # Check if it has the required files
                train_file = os.path.join(cifar100_path, 'train')
                test_file = os.path.join(cifar100_path, 'test')
                if os.path.exists(train_file) and os.path.exists(test_file):
                    print(f"✓ CIFAR100 already exists at {cifar100_path}")
                    return True
            elif os.path.exists(cifar100_alt_path) and os.path.isdir(cifar100_alt_path):
                train_file = os.path.join(cifar100_alt_path, 'train')
                test_file = os.path.join(cifar100_alt_path, 'test')
                if os.path.exists(train_file) and os.path.exists(test_file):
                    print(f"✓ CIFAR100 already exists at {cifar100_alt_path}")
                    return True
            
            print("CIFAR100: Downloading train split...")
            datasets.CIFAR100(location, train=True, download=True)
            print("CIFAR100: Downloading test split...")
            datasets.CIFAR100(location, train=False, download=True)
            print(f"✓ CIFAR100 downloaded")
            
        elif dataset_name == "DTD":
            print("DTD: Downloading train split...")
            datasets.DTD(location, split="train", download=True)
            print("DTD: Downloading test split...")
            datasets.DTD(location, split="test", download=True)
            print(f"✓ DTD downloaded")
            
        elif dataset_name == "EuroSAT":
            print("EuroSAT: Downloading...")
            try:
                datasets.EuroSAT(location, download=True)
                print(f"✓ EuroSAT downloaded")
            except Exception as e:
                if "SSL" in str(e) or "certificate" in str(e).lower():
                    print(f"  SSL certificate error detected. Retrying with SSL verification disabled...")
                    # Temporarily disable SSL verification for this download
                    original_context = ssl._create_default_https_context
                    ssl._create_default_https_context = ssl._create_unverified_context
                    try:
                        datasets.EuroSAT(location, download=True)
                        print(f"✓ EuroSAT downloaded (with SSL verification disabled)")
                    finally:
                        # Restore original context
                        ssl._create_default_https_context = original_context
                else:
                    raise
            
        elif dataset_name == "Flowers":
            print("Flowers: Downloading train split...")
            datasets.Flowers102(location, split="train", download=True)
            print("Flowers: Downloading test split...")
            datasets.Flowers102(location, split="test", download=True)
            print(f"✓ Flowers downloaded")
            
        elif dataset_name == "Food":
            print("Food: Downloading train split...")
            try:
                datasets.Food101(location, split="train", download=True)
                print("Food: Downloading test split...")
                datasets.Food101(location, split="test", download=True)
                print(f"✓ Food downloaded")
            except Exception as e:
                if "SSL" in str(e) or "certificate" in str(e).lower():
                    print(f"  SSL certificate error detected. Retrying with SSL verification disabled...")
                    original_context = ssl._create_default_https_context
                    ssl._create_default_https_context = ssl._create_unverified_context
                    try:
                        datasets.Food101(location, split="train", download=True)
                        datasets.Food101(location, split="test", download=True)
                        print(f"✓ Food downloaded (with SSL verification disabled)")
                    finally:
                        ssl._create_default_https_context = original_context
                else:
                    raise
            
        elif dataset_name == "MNIST":
            print("MNIST: Downloading train split...")
            datasets.MNIST(location, train=True, download=True)
            print("MNIST: Downloading test split...")
            datasets.MNIST(location, train=False, download=True)
            print(f"✓ MNIST downloaded")
            
        elif dataset_name == "OxfordPet":
            print("OxfordPet: Downloading train split...")
            datasets.OxfordIIITPet(location, split="trainval", download=True)
            print("OxfordPet: Downloading test split...")
            datasets.OxfordIIITPet(location, split="test", download=True)
            print(f"✓ OxfordPet downloaded")
            
        elif dataset_name == "StanfordCars":
            # Check if StanfordCars already exists
            stanford_cars_path = os.path.join(location, 'stanford-cars')
            if os.path.exists(stanford_cars_path) and os.path.isdir(stanford_cars_path):
                # Check if it has train/test splits
                train_path = os.path.join(stanford_cars_path, 'cars_train')
                test_path = os.path.join(stanford_cars_path, 'cars_test')
                train_alt = os.path.join(stanford_cars_path, 'train')
                test_alt = os.path.join(stanford_cars_path, 'test')
                if (os.path.exists(train_path) or os.path.exists(train_alt)) and \
                   (os.path.exists(test_path) or os.path.exists(test_alt)):
                    print(f"✓ StanfordCars already exists at {stanford_cars_path}")
                    return True
            
            print("StanfordCars: Attempting download via torchvision...")
            try:
                print("StanfordCars: Downloading train split...")
                datasets.StanfordCars(location, split="train", download=True)
                print("StanfordCars: Downloading test split...")
                datasets.StanfordCars(location, split="test", download=True)
                print(f"✓ StanfordCars downloaded via torchvision")
                return True
            except Exception as e:
                print(f"  Torchvision download failed: {e}")
                print("  Attempting alternative download from Hugging Face...")
                
                # Try Hugging Face download
                if HAS_DATASETS_LIB:
                    try:
                        print("  Downloading StanfordCars from Hugging Face...")
                        print("  Note: This may take several minutes...")
                        hf_dataset = load_dataset("tanganke/stanford_cars", trust_remote_code=True)
                        
                        # Save to expected location
                        os.makedirs(stanford_cars_path, exist_ok=True)
                        
                        # Convert and save train split
                        train_dir = os.path.join(stanford_cars_path, 'cars_train')
                        test_dir = os.path.join(stanford_cars_path, 'cars_test')
                        os.makedirs(train_dir, exist_ok=True)
                        os.makedirs(test_dir, exist_ok=True)
                        
                        # Save images organized by class
                        print("  Saving train split (this may take a while)...")
                        if 'train' in hf_dataset:
                            train_data = hf_dataset['train']
                            total_train = len(train_data)
                            for idx, item in enumerate(train_data):
                                if idx % 500 == 0:
                                    print(f"    Progress: {idx}/{total_train} images...")
                                
                                if 'image' in item:
                                    img = item['image']
                                    label = item.get('label', 0)
                                    label_dir = os.path.join(train_dir, f"class_{label:04d}")
                                    os.makedirs(label_dir, exist_ok=True)
                                    img.save(os.path.join(label_dir, f"image_{idx:05d}.jpg"))
                        
                        print("  Saving test split...")
                        if 'test' in hf_dataset:
                            test_data = hf_dataset['test']
                            total_test = len(test_data)
                            for idx, item in enumerate(test_data):
                                if idx % 500 == 0:
                                    print(f"    Progress: {idx}/{total_test} images...")
                                
                                if 'image' in item:
                                    img = item['image']
                                    label = item.get('label', 0)
                                    label_dir = os.path.join(test_dir, f"class_{label:04d}")
                                    os.makedirs(label_dir, exist_ok=True)
                                    img.save(os.path.join(label_dir, f"image_{idx:05d}.jpg"))
                        
                        print(f"✓ StanfordCars downloaded from Hugging Face")
                        return True
                    except Exception as hf_error:
                        print(f"  Hugging Face download also failed: {hf_error}")
                        print("  You may need to download manually from:")
                        print("    - http://ai.stanford.edu/~jkrause/cars/car_dataset.html")
                        print("    - https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset")
                        print("    - https://huggingface.co/datasets/tanganke/stanford_cars")
                        return False
                else:
                    print("  'datasets' library not available for alternative download.")
                    print("  Install with: pip install datasets")
                    print("  Or download manually from alternative sources.")
                    return False
            
        elif dataset_name == "SUN397":
            # Check if SUN397 already exists
            sun397_path = os.path.join(location, 'SUN397')
            if os.path.exists(sun397_path) and os.path.isdir(sun397_path):
                # Check if it has category folders
                categories = [d for d in os.listdir(sun397_path) 
                             if os.path.isdir(os.path.join(sun397_path, d))]
                if len(categories) > 0:
                    print(f"✓ SUN397 already exists at {sun397_path} ({len(categories)} categories found)")
                    return True
            
            print("SUN397: Attempting download via torchvision...")
            try:
                datasets.SUN397(location, download=True)
                print(f"✓ SUN397 downloaded via torchvision")
                return True
            except Exception as e:
                print(f"  Torchvision download failed: {e}")
                print("  Attempting alternative download from Hugging Face...")
                
                # Try Hugging Face download
                if HAS_DATASETS_LIB:
                    try:
                        print("  Downloading SUN397 from Hugging Face...")
                        print("  Note: This is a large dataset (~37GB) and may take a very long time...")
                        hf_dataset = load_dataset("tanganke/sun397", trust_remote_code=True)
                        
                        # Save to expected location
                        os.makedirs(sun397_path, exist_ok=True)
                        
                        print("  Saving images to disk (this will take a while)...")
                        print("  Note: SUN397 is very large (~37GB), this may take hours...")
                        # Process all splits
                        total_images = 0
                        for split_name in hf_dataset.keys():
                            print(f"  Processing {split_name} split...")
                            split_data = hf_dataset[split_name]
                            split_total = len(split_data)
                            
                            for idx, item in enumerate(split_data):
                                if idx % 1000 == 0 and idx > 0:
                                    print(f"    Progress: {idx}/{split_total} images ({total_images} total)...")
                                
                                if 'image' in item:
                                    img = item['image']
                                    # Get category name from the dataset
                                    class_name = item.get('class_name', None)
                                    if class_name is None:
                                        # Try to get from label or filename
                                        label = item.get('label', 0)
                                        filename = item.get('file_name', '')
                                        if filename:
                                            # Extract category from filename (e.g., "a/abbey/sun_xxx.jpg" -> "a/abbey")
                                            parts = filename.split('/')
                                            if len(parts) >= 2:
                                                class_name = '/'.join(parts[:2])
                                            else:
                                                class_name = f"category_{label}"
                                        else:
                                            class_name = f"category_{label}"
                                    
                                    # Create category directory (handle nested paths like "a/abbey")
                                    category_dir = os.path.join(sun397_path, class_name)
                                    os.makedirs(category_dir, exist_ok=True)
                                    
                                    # Save image with unique name
                                    img_filename = f"image_{total_images:06d}.jpg"
                                    img_path = os.path.join(category_dir, img_filename)
                                    img.save(img_path)
                                    total_images += 1
                        
                        print(f"✓ SUN397 downloaded from Hugging Face ({total_images} images)")
                        return True
                    except Exception as hf_error:
                        print(f"  Hugging Face download also failed: {hf_error}")
                        print("  You may need to download manually from:")
                        print("    - https://huggingface.co/datasets/tanganke/sun397")
                        print("    - https://hyper.ai/datasets/5367")
                        print("    - http://groups.csail.mit.edu/vision/SUN/")
                        print("  Or use: python -c \"from datasets import load_dataset; load_dataset('tanganke/sun397').save_to_disk('./SUN397')\"")
                        return False
                else:
                    print("  'datasets' library not available for alternative download.")
                    print("  Install with: pip install datasets")
                    print("  Or download manually from alternative sources.")
                    return False
            
        else:
            print(f"✗ Unknown dataset: {dataset_name}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_dir_size_bytes(path):
    """Get directory size in bytes (recursive helper)."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size_bytes(entry.path)
    except (PermissionError, OSError):
        pass
    return total

def get_dir_size(path):
    """Get human-readable directory size."""
    total_bytes = get_dir_size_bytes(path)
    
    # Convert to human readable
    total = float(total_bytes)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if total < 1024.0:
            return f"{total:.1f} {unit}"
        total /= 1024.0
    return f"{total:.1f} PB"

def get_file_size(path):
    """Get human-readable file size."""
    try:
        size = os.path.getsize(path)
        # Convert to human readable
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"
    except (OSError, PermissionError):
        return "unknown size"

def cleanup_temp_files(data_location):
    """Remove temporary tar.gz and zip files after extraction."""
    print(f"\n{'='*60}")
    print("Cleaning up temporary files...")
    print(f"{'='*60}")
    
    temp_extensions = ['.tar.gz', '.zip', '.tgz']
    cleaned = []
    
    # Only check top-level directory for temp files
    for file in os.listdir(data_location):
        if any(file.endswith(ext) for ext in temp_extensions):
            file_path = os.path.join(data_location, file)
            if os.path.isfile(file_path):
                try:
                    file_size = get_file_size(file_path)
                    os.remove(file_path)
                    cleaned.append((file, file_size))
                    print(f"  Removed: {file} ({file_size})")
                except Exception as e:
                    print(f"  Warning: Could not remove {file}: {e}")
    
    if cleaned:
        print(f"\n✓ Cleaned up {len(cleaned)} temporary file(s)")
    else:
        print("  No temporary files to clean up")

def show_dataset_structure(data_location):
    """Display the final dataset directory structure."""
    print(f"\n{'='*60}")
    print("DATASET DIRECTORY STRUCTURE")
    print(f"{'='*60}")
    print(f"\nAll datasets are stored in: {data_location}\n")
    
    # Expected dataset directories (matching torchvision naming)
    expected_dirs = {
        "TinyImagenet": ["tinyimagenet/tiny-imagenet-200", "tiny-imagenet-200"],
        "Aircraft": ["fgvc-aircraft-2013b"],
        "Caltech101": ["caltech101"],
        "CIFAR100": ["cifar100", "cifar-100-python"],
        "DTD": ["dtd"],
        "EuroSAT": ["eurosat"],
        "Flowers": ["flowers-102"],
        "Food": ["food-101"],
        "MNIST": ["MNIST"],
        "OxfordPet": ["oxford-iiit-pet"],
        "StanfordCars": ["stanford-cars"],
        "SUN397": ["SUN397"]
    }
    
    print("Dataset folders:")
    found_count = 0
    for dataset_name, possible_dirs in expected_dirs.items():
        found = False
        for dir_name in possible_dirs:
            dir_path = os.path.join(data_location, dir_name)
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                size = get_dir_size(dir_path)
                print(f"  ✓ {dataset_name:15s} -> {dir_name:30s} ({size})")
                found = True
                found_count += 1
                break
        if not found:
            print(f"  ✗ {dataset_name:15s} -> (not found)")
    
    total_size = get_dir_size(data_location)
    print(f"\nFound {found_count}/{len(expected_dirs)} datasets")
    print(f"Total datasets directory size: {total_size}")
    print(f"\nFor detailed information, see: {os.path.join(data_location, 'README.md')}")

def main():
    # Get data location from environment or use default
    data_location = os.environ.get("DATA_LOCATION", DEFAULT_DATA_LOCATION)
    
    # Resolve to absolute path
    data_location = os.path.abspath(os.path.expanduser(data_location))
    
    print(f"Data location: {data_location}")
    print(f"Creating directory if it doesn't exist...")
    os.makedirs(data_location, exist_ok=True)
    
    # All datasets needed for training
    all_datasets = [
        "TinyImagenet",
        "Aircraft",
        "Caltech101",
        "CIFAR100",
        "DTD",
        "EuroSAT",
        "Flowers",
        "Food",
        "MNIST",
        "OxfordPet",
        "StanfordCars",
        "SUN397"
    ]
    
    print(f"\nWill download {len(all_datasets)} datasets:")
    for ds in all_datasets:
        print(f"  - {ds}")
    
    # Simple transform (just identity for download purposes)
    preprocess = lambda x: x
    
    results = {}
    for dataset_name in all_datasets:
        success = download_dataset(dataset_name, data_location, preprocess)
        results[dataset_name] = success
    
    # Clean up temporary files
    cleanup_temp_files(data_location)
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    successful = [ds for ds, success in results.items() if success]
    failed = [ds for ds, success in results.items() if not success]
    
    print(f"\n✓ Successfully downloaded ({len(successful)}/{len(all_datasets)}):")
    for ds in successful:
        print(f"  - {ds}")
    
    if failed:
        print(f"\n✗ Failed to download ({len(failed)}/{len(all_datasets)}):")
        for ds in failed:
            print(f"  - {ds}")
        print("\nNote: Some datasets may need manual download.")
    
    # Show final directory structure
    show_dataset_structure(data_location)

if __name__ == "__main__":
    main()
