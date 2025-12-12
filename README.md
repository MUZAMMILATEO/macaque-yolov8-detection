### Environment setup

Create base environment:
```bash
conda env create -f environment.yml
conda activate macaque-task
```

Install PyTorch (choose one):
```bash
# GPU (CUDA 11.8)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

Run the following command to download and organise the data
```bash
python setup_data.py
```

This command will arrange the downloaded data in the following structure
```bash
macaque-dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── test/
│   ├── images/
│   └── labels/
└── macaque_data.yaml
```
__Note:__ The test set is stored at the root level (macaque-dataset/test/)
and is intentionally separated from the YOLO train/val structure.

## Training and Validation
To train the YOLOv8 model and perform validation on the macaque dataset, run:
```bash
python train_model.py
```
This script:
- Converts the dataset labels to YOLO-compatible format (5-column annotations)
- Prepares a YOLO-view dataset for training and validation
- Trains a YOLOv8 model using transfer learning
- Evaluates performance on the validation split
- Saves model checkpoints and logs under ./root/runs/detect/