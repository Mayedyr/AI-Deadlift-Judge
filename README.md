# AI-Deadlift-Judge
The system uses a YOLO11 pose estimation model that extracts joint key points from each video frame. These key points are standardized through a preprocessing pipeline and fed into a custom TCN, which outputs predictions for three distinct infraction cards as defined by the IPF.

I built this project for my final-year university project. Feel free to explore it or use any part of it if you find it useful.

Run DLJudgeApp.py for the UI and inference interface.
LoadVideoData.ipynb processes raw videos into training-ready tensors.
DLJudgeTrain.ipynb trains the model and produces model_weights.pt.

Refer to PDF_DLJudge_Report.pdf for the full details on design, methodology, implementation, evaluation, and discussion.
