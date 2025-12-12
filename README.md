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