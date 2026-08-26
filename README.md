# AI Deadlift Judge

AI Deadlift Judge is a computer-vision system for detecting technical rule violations in powerlifting deadlift attempts from video. It uses YOLO11 pose estimation to extract 17 body keypoints per frame and a Temporal Convolutional Network (TCN) to classify three International Powerlifting Federation (IPF) deadlift infraction-card categories.

This repository contains the preprocessing notebook, TCN training and evaluation notebook, pretrained TCN weights, and a Streamlit inference application developed for the project.

> The original project report is available in [`PDF_DLJudge_Report.pdf`](PDF_DLJudge_Report.pdf).

## Method

The pipeline is:

`Deadlift video -> YOLO11 pose estimation -> 17 x 2 keypoints per frame -> pad/truncate to 200 frames -> 34 x 200 pose sequence -> TCN -> Red / Blue / Yellow card predictions`

The three outputs are treated as independent binary labels because one attempt can contain more than one technical fault.

| Output | Model interpretation |
| --- | --- |
| Red card | Soft/incomplete lockout |
| Blue card | Downward movement or support on the thighs |
| Yellow card | Other technical infraction or incomplete lift |

A lift is returned as a **Good Lift** only when none of the three card outputs exceed the classification threshold.

### TCN configuration

| Setting | Value |
| --- | --- |
| Input | 17 two-dimensional keypoints, 34 channels |
| Sequence length | 200 frames |
| TCN channels | 50, 75, 100, 125, 125 |
| Kernel size | 7 |
| Dropout | 0.5 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Training epochs | 400 |
| Classification threshold | 0.9 |
| Loss | BCE with logits and positive class weights |

Training also uses weighted random sampling to reduce the effect of class imbalance.

## Reported results

The project report gives the following validation results for the full judging pipeline:

| Metric | YOLO11x-pose | YOLO11n-pose |
| --- | ---: | ---: |
| Exact-match accuracy | 80.00% | 76.67% |
| Weighted precision | 74.18% | 77.95% |
| Weighted recall | 92.31% | 76.92% |
| Weighted F1 | 82.15% | 76.78% |
| Judge score | 83.33% | 80.00% |

The judge score compares the final **Good Lift / No Lift** decision with the labelled decision, while exact-match accuracy requires the complete three-card output vector to match.

End-to-end evaluation time measured on an NVIDIA RTX 4060 was:

| Time | YOLO11x-pose | YOLO11n-pose |
| --- | ---: | ---: |
| Average | 5.67 s | 2.33 s |
| Maximum | 9.91 s | 3.78 s |

The current [`DLJudgeApp.py`](DLJudgeApp.py) is configured to use `yolo11x-pose.pt`.

## Repository structure

| File | Purpose |
| --- | --- |
| [`LoadVideoData.ipynb`](LoadVideoData.ipynb) | Extracts YOLO11 pose keypoints from raw videos, standardizes each sequence to 200 frames, applies horizontal mirroring augmentation, and saves the processed tensor as `all_videos.pt`. |
| [`DLJudgeTrain.ipynb`](DLJudgeTrain.ipynb) | Loads processed pose sequences and labels, trains the TCN, evaluates it, and saves the trained weights. |
| [`DLJudgeApp.py`](DLJudgeApp.py) | Streamlit application for running inference on an uploaded MP4 video. |
| [`model_weights.pt`](model_weights.pt) | Pretrained TCN weights used by the inference application. |
| [`PDF_DLJudge_Report.pdf`](PDF_DLJudge_Report.pdf) | Original project report with the full methodology, implementation, evaluation, and discussion. |
| [`LICENSE`](LICENSE) | MIT licence for this repository. |

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/Mayedyr/AI-Deadlift-Judge.git
cd AI-Deadlift-Judge
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

Install the packages used by the project:

```bash
pip install torch ultralytics pytorch-tcn streamlit pandas scikit-learn jupyter
```

Ultralytics will obtain the required YOLO11 pose weights when the model is first loaded if they are not already available locally.

## Running inference

The pretrained TCN weights are included in the repository. Start the Streamlit application with:

```bash
streamlit run DLJudgeApp.py
```

Then upload an MP4 deadlift video through the browser interface. The application:

1. extracts normalized pose keypoints with YOLO11-pose;
2. pads or truncates the sequence to 200 frames;
3. reshapes the sequence to `34 x 200`;
4. applies the pretrained TCN;
5. returns any detected card categories and the derived Good Lift / No Lift decision.

CUDA is used automatically when available. Otherwise, PyTorch runs on the CPU.

## Reproducing preprocessing and training

The preprocessing and training notebooks expect the original dataset in a local `DLdataset/` directory. The raw training data are **not included in this public repository**.

Expected layout:

```text
AI-Deadlift-Judge/
├── DLdataset/
│   ├── 1.mp4
│   ├── 2.mp4
│   ├── ...
│   └── simple_labels.csv
├── LoadVideoData.ipynb
├── DLJudgeTrain.ipynb
└── ...
```

`simple_labels.csv` is expected to contain an `ID` column and binary `Red`, `Blue`, and `Yellow` columns corresponding to the three card outputs.

The intended reproduction sequence is:

1. Place the numbered deadlift videos and `simple_labels.csv` in `DLdataset/`.
2. Run [`LoadVideoData.ipynb`](LoadVideoData.ipynb) to create `all_videos.pt`.
3. Run [`DLJudgeTrain.ipynb`](DLJudgeTrain.ipynb) to train and evaluate the TCN.
4. Save the resulting model state as `model_weights.pt` for use by the inference application.

The preprocessing notebook horizontally mirrors the pose sequences to augment the training data. The reported dataset contains 375 source videos and 750 pose sequences after this augmentation.

## Data availability

The repository currently includes the project code, pretrained TCN weights, and project report. It does **not** include:

- the original deadlift video dataset;
- `DLdataset/simple_labels.csv`;
- the generated `all_videos.pt` training tensor.

These files are therefore required separately to reproduce the original training run. The inference application can be used without them because [`model_weights.pt`](model_weights.pt) is included.

## Limitations

The system is a research prototype rather than a replacement for competition referees. Important limitations include the relatively small and imbalanced training set, sensitivity of two-dimensional pose estimation to camera angle and occlusion, and the fact that body pose alone does not directly encode the bar path, bar-to-thigh contact, grip control, or referee command timing used in some IPF rules.

## Citation

If you use this repository in academic work, please cite the accompanying project report until a conference-paper citation is available:

```text
Majid Alredha, "AI Deadlift Judge: Automated Detection of Technical Rule Violations in Powerlifting," final-year project report, University of Birmingham Dubai, 2025.
```

## Licence

The repository code is available under the [MIT License](LICENSE). Third-party packages and pretrained YOLO models are subject to their respective licences.
