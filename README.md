## [WACV 2025 Oral] [Transferring Foundation Models for Generalizable Robotic Manipulation](https://arxiv.org/pdf/2306.05716)
## [Arxiv 23.06] [Pave the Way to Grasp Anything: Transferring Foundation Models for Universal Pick-Place Robots](https://arxiv.org/abs/2306.05716v1)
![caps](./frame_work.png)

## Getting started

### Preparation:
0. Install ROS Noetic

1. Clone GroundingDINO for open-vocab object detection
`git clone https://github.com/IDEA-Research/GroundingDINO.git`

2. Clone MixFormer for object tracking:
`git clone https://github.com/MCG-NJU/MixFormer.git`

3. Clone Segment-Anything for segmentation:
`https://github.com/facebookresearch/segment-anything.git`

### Train the policy model:
`
python ./model/train.py
`

### Run Mnipulation System:
1. Launch the detection, tracking and segmentation services.
2. Launch the policy model
`python ./model/grasp_service.py`
