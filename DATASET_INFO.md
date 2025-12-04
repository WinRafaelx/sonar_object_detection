# Dataset Download Information

## Where Datasets Are Downloaded

### Roboflow Downloads

When using Roboflow to download datasets, the dataset is downloaded to the **current working directory** with the following naming convention:

```
{project_name}-{version_number}/
```

**Example:**
- Project: `sonar_detect`
- Version: `1`
- Download location: `sonar_detect-1/`

### Current Dataset Location

Based on your project structure, the dataset is located at:
```
sonar_detect-1/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The `data.yaml` file contains the dataset configuration and paths.

## How to Download Datasets

### Option 1: Using train_sonar.py (Recommended)

The updated `train_sonar.py` script now supports automatic dataset downloading:

```bash
python train_sonar.py --download-roboflow
```

This will:
1. Download the dataset from Roboflow to the current directory
2. Automatically use the downloaded dataset for training

**Custom Roboflow settings:**
```bash
python train_sonar.py --download-roboflow \
    --roboflow-workspace object-detect-ury2h \
    --roboflow-project sonar_detect \
    --roboflow-version 1
```

### Option 2: Using Original train.py

The original `train.py` script downloads the dataset:

```bash
python train.py
```

**Requirements:**
- Set `ROBOFLOW_API_KEY` environment variable:
  ```bash
  # Windows
  set ROBOFLOW_API_KEY=your_api_key
  
  # Linux/Mac
  export ROBOFLOW_API_KEY=your_api_key
  ```

### Option 3: Manual Download

If you already have the dataset downloaded, specify the path:

```bash
python train_sonar.py --data sonar_detect-1/data.yaml
```

## Dataset Structure

The downloaded dataset follows YOLO format:

```
{project}-{version}/
├── data.yaml          # Dataset configuration
├── README.roboflow.txt # Roboflow metadata
├── train/              # Training split
│   ├── images/         # Training images
│   └── labels/         # Training labels (YOLO format)
├── valid/              # Validation split
│   ├── images/
│   └── labels/
└── test/               # Test split
    ├── images/
    └── labels/
```

## data.yaml Contents

The `data.yaml` file typically contains:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 4  # Number of classes
names: ['aircraft', 'fish', 'other', 'shipwreck']

roboflow:
  workspace: object-detect-ury2h
  project: sonar_detect
  version: 1
  license: CC BY 4.0
```

## Notes

1. **Download Location:** Roboflow always downloads to the current working directory where the script is run.

2. **Existing Datasets:** If a dataset folder already exists, Roboflow may skip downloading or prompt for confirmation.

3. **API Key:** You need a Roboflow account and API key. Get it from: https://roboflow.com/

4. **Git Ignore:** The `sonar_detect-1/` folder is typically in `.gitignore` to avoid committing large datasets.

5. **Path Resolution:** The `data.yaml` uses relative paths (`../train/images`), so make sure to run training from the correct directory or update the paths.

## Troubleshooting

**Problem:** "ROBOFLOW_API_KEY not set"
- Solution: Set the environment variable with your Roboflow API key

**Problem:** "Dataset not found"
- Solution: Either download using `--download-roboflow` or specify path with `--data`

**Problem:** "Cannot find data.yaml"
- Solution: Check that the dataset folder exists and contains `data.yaml`

