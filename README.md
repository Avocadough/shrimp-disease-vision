# Shrimp image detection experiments

An exploratory computer-vision project using YOLO to detect three historical model labels in shrimp images: `BG`, `WSSV` and `WSSV_BG`. The repository contains training notebooks, a Streamlit interface, historical training outputs and checkpoints.

> This is a research prototype, not a veterinary diagnosis tool. A model score or a missing detection does not establish whether an animal is healthy. Dataset provenance, labeling quality, independent validation and generalization to new farms or cameras are not established here.

## Explore the project

| Path | Purpose |
| --- | --- |
| [`shrimp_detec.ipynb`](shrimp_detec.ipynb) | Original experiments and saved training/evaluation output. |
| [`new_train.ipynb`](new_train.ipynb) | Additional dataset-preparation experiment. |
| [`app_v2.py`](app_v2.py) | Streamlit prototype for uploading a single image or using a camera. |
| [`runs/detect/shrimp_disease_best/weights/best.pt`](runs/detect/shrimp_disease_best/weights/best.pt) | Historical checkpoint used by the interface. |
| [`runs/detect/shrimp_disease_best/`](runs/detect/shrimp_disease_best/) | Saved training configuration, metrics and checkpoints. |

The notebooks reference local data and model paths from the original machine; the full training data is not included. This repository also contains many large historical checkpoint files, so a full clone is sizable.

## Start the interface

The app uses the included checkpoint at `runs/detect/shrimp_disease_best/weights/best.pt` relative to the repository root. With Python and compatible dependencies:

```bash
python -m pip install -r requirements.txt
streamlit run app_v2.py
```

The UI startup has been checked locally with Python 3.10, Streamlit 1.35 and Ultralytics 8.3.168. End-to-end image inference has **not** been verified in that environment because its installed Torch and torchvision versions are incompatible. A clean environment with matching versions is needed. The Dockerfile has been corrected to reference the checked-in application and checkpoint, but the container has not been built or tested.

To check the UI and simulated upload flow, install `pytest` and run `python -m pytest -q`. The upload tests use a mocked model; they do not test model accuracy or full image inference. Load PyTorch checkpoints only when you trust their source.

## Evaluation limits

The notebook contains multiple experiments and saved metrics. They should not be presented as an independent benchmark until the underlying dataset, split and exact checkpoint are traced and reproduced. A credible assessment requires provenance, suitable held-out groups and independently reviewed labels.
