# PPG Arrhythmia Detection

This package trains multi-class arrhythmia detection models using PPG signals stored as paired `.mat`/`.hea` records (channel 2 = PLETH). It provides:

- Data loaders for PhysioNet-style files.
- Preprocessing with filtering, normalization, segmentation, and augmentation.
- Feature-based (RandomForest) and deep learning (CNN + BiLSTM) classifiers.
- Training pipeline producing metrics and persisted weights.
- Inference helpers and an end-to-end example script.

## Quick Start

```bash
python -m pip install -r requirements.txt
python train.py --data-root ../training --output weights
python inference.py path/to/window.npy --weights weights --plot
```

The `examples/run_example.py` script walks through training and inference on a random sample in the dataset.
