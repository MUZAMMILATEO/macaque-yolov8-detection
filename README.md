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
_Note:_ The test set is stored at the root level (macaque-dataset/test/)
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

## Inference

To run inference using the trained YOLOv8 model, use:

```bash
python infer_model.py [image_name]
```

- Place the image to be tested inside the `sample_image/` folder.
- The image name can be provided with or without an extension (.jpg, .jpeg, .png).

### Example
```bash
python infer_model.py
python infer_model.py test.jpg
python infer_model.py test
```
_Note:_ If no image name is provided, the script defaults to sample_image/sample.jpg.


## Test Evaluation

After training, evaluate the `latest trained model` on the held-out test split using:

```bash
python eval_model.py
```

The script automatically selects the most recent training run from runs/detect/ (e.g. macaque_detection_run, macaque_detection_run2, macaque_detection_run3, ...).

**Outputs:** Evaluation results (metrics + plots) are saved under `runs/detect/macaque_test_evaluation/<N>/`, where <N> matches the suffix of the training run used (e.g., if macaque_detection_run3 is used, results are saved to .../3/).
