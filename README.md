# Efficient Track Anything#
[[`📕Project`](https://yformer.github.io/efficient-track-anything/)][[`🤗Gradio Demo`](https://10f00f01361a8328a4.gradio.live)][[`📕Paper`](https://arxiv.org/pdf/2411.18933)][[`🤗Checkpoints`]](https://huggingface.co/yunyangx/efficient-track-anything/tree/main)[['📕Project '] (https://yformer.github.io/efficient-track-anything/)][[🤗Gradio演示的](https://10f00f01361a8328a4.gradio.live)][[‘📕Paper] (https://arxiv.org/pdf/2411.18933)][['🤗Checkpoints ']] (https://huggingface.co/yunyangx/efficient-track-anything/tree/main)

Run Efficient Track Anything on a live video stream


## EfficientTAM Video Segmentation Examples
  |   |   |
:-------------------------:|:-------------------------:
SAM 2 [SAM2]（figs/examples/sam2_video_segmentation.png）| ![SAM2](figs/examples/sam2_video_segmentation.png无花果/ / sam2_video_segmentation.png例子)
EfficientTAM EfficientTAM （figs/examples/efficienttam_video_segmentation.png）|  ![EfficientTAM](figs/examples/efficienttam_video_segmentation.png无花果/ / efficienttam_video_segmentation.png例子)

## EfficientTAM Image Segmentation Examples
Input Image, SAM, EficientSAM, SAM 2, EfficientTAM
  |   |   |
:-------------------------:|:-------------------------:
Point-prompt [Point-prompt]（figs/examples/demo_img_point.png）| ![point-prompt](figs/examples/demo_img_point.png无花果/ / demo_img_point.png例子)[Point-prompt](figs/examples/demo_img_point.png)| ![Point-prompt]（figs/examples/demo_img_point.png）
Box-prompt [Box-prompt]（figs/examples/demo_img_box.png）|  ![box-prompt](figs/examples/demo_img_box.png无花果/ / demo_img_box.png例子)[Box-prompt](figs/examples/demo_img_box.png)| ![Box-prompt]（figs/examples/demo_img_box.png）
Segment everything 分割所有东西，b| ！(段一切)(无花果/ / demo_img_everything.png例子)|![segment everything   段的一切](figs/examples/demo_img_everything.png无花果/ / demo_img_everything.png例子)

## Model  
EfficientTAM checkpoints are available at the 在[拥抱空间]（https://huggingface.co/yunyangx/efficient-track-anything/tree/main）上可以找到有效的tam检查点。[Hugging Face Space   拥抱的脸部空间](https://huggingface.co/yunyangx/efficient-track-anything/tree/main).

## Getting Started  

### Installation 

```bash   ”“bash   ”“bash”“bash
git clone https://github.com/yformer/EfficientTAM.git
cd EfficientTAM
conda create -n efficient_track_anything python=3.12Conda create -n efficient_track_anything python=3.12conda create -n efficient_track_anything python=3.12 conda create -n efficient_track_anything python=3.12
conda activate efficient_track_anything激活efficient_track_anything激活efficient_track_anything
pip install -e .   PIP安装-e。PIP安装-e。PIP。
```
### Download Checkpoints

```bash   ”“bash   ”“bash”“bash“bash”“bash”“bash”“bash”
cd checkpoints   cd检查点
./download_checkpoints.sh
```

We can benchmark FPS of efficient track anything models on GPUs and model size.我们可以对FPS进行基准测试，有效地跟踪gpu和模型尺寸上的任何模型。

### FPS Benchmarking and Model Size

```bash   ”“bash
cd ..
python efficient_track_anything/benchmark.py
```

### Launching Gradio Demo Locally
For efficient track anything video, run为了有效地跟踪任何视频，运行
```
python app.py
```
For efficient track anything image, run为了有效地跟踪任何图像，运行
```
python app_image.py
```


### Building Efficient Track Anything
You can build efficient track anything model with a config and initial the model with a checkpoint,你可以用配置建立有效的跟踪任何模型，并用检查点初始化模型，
```python   ”“python   ”“python”“python“蟒蛇” “蟒蛇” “蟒蛇” "蟒蛇
import torch   进口火炬

from efficient_track_anything.build_efficienttam import (从efficient_track_anything。Build_efficienttam导入
    build_efficienttam_video_predictor,
)

checkpoint = "./checkpoints/efficienttam_s.pt"检查点= “./检查点/efficienttam_s.pt”
model_cfg = "configs/efficienttam/efficienttam_s.yaml"model_cfg = “configs/efficienttam/efficienttam_s.yaml”model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = “ configs/efficienttam/efficienttam_s.yaml ”model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml”model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml“model_cfg = ”configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml model_cf”configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"mog = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml mo”del_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。那里del_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttamam_s。yaml"model_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/ efficientam /effil"model_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/ efficientam /efficientam_s。yaml"model_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configmodel_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml " model_cfg = "configs/efficienttam/efficienttam_s。yaml"model_cfg = " configs/efficienttam/efficienttam_s。yaml”

predictor = build_efficienttam_video_predictor(model_cfg, checkpoint)Predictor = build_efficienttam_video_predictor（model_cfg，检查点）predictor = build_efficienttam_video_predictor（model_cfg, checkpoint）predictor = build_efficienttam_video_predictor(model_cfg, checkpoint) predictor = build_efficienttam_video_predictor（model_cfg, checkpoint）predictor = build_efficienttam_video_predictor(model_cfg, checkpoint)predictor = build_efficienttam_video_predictor(model_cfg, checkpoint)predictor = build_efficienttam_videopredictor = build_efficienttam_video_predictor（model_cfg, checkpoint）)predictor = build_efficienttam_video_predictor（model_cfg，检查点）predictor = build_efficienttam_video_predictor（model_cfg，检查点）predictor = build_efficienttam_video_predictor（model_cfg，检查点）predictor = build_efficienttam_video_predictor（model_cfg，检查点）predictor = build_efficienttam_video_predictor（model_cfg，检查点）predictor = build_efficienttam_video_predictor（model_cfg，检查点）predictor = build_efficienttam_video_video_predictor(model_cfg, checkpoint)predictor = build_efficienttam_video_predictor（model_cfg, checkpoint）
```

### Efficient Track Anything Notebook Example
The notebook is shared [here](https://github.com/yformer/EfficientTAM/blob/main/notebooks)笔记本在这里分享（https://github.com/yformer/EfficientTAM/blob/mainThe notebook is shared [here](https://github.com/yformer/EfficientTAM/blob/main
otebooks)笔记本在这里分享（https://github.com/yformer/EfficientTAM/blob/main
otebooks）   otebooks)

### Efficient Track Anything in stream mode

```
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor

checkpoint = "../checkpoints/efficienttam_s.pt"
model_cfg = "configs/efficienttam/efficienttam_s.yaml"

predictor = build_efficienttam_camera_predictor(model_cfg, checkpoint)
```

## Reference
Efficient Track Anything(https://github.com/yformer/EfficientTAM)

SAM2-real-time(https://github.com/Gy920/segment-anything-2-real-time/tree/main)
