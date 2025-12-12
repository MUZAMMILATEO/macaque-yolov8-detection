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