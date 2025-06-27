# Efficient Track Anything#高效跟踪   高效跟踪
[[`📕Project`](https://yformer.github.io/efficient-track-anything/)][[`🤗Gradio Demo`](https://10f00f01361a8328a4.gradio.live)][[`📕Paper`](https://arxiv.org/pdf/2411.18933)][[`🤗Checkpoints`]](https://huggingface.co/yunyangx/efficient-track-anything/tree/main)[['📕Project '] (https://yformer.github.io/efficient-track-anything/)][[🤗Gradio演示的](https://10f00f01361a8328a4.gradio.live)][[‘📕Paper] (https://arxiv.org/pdf/2411.18933)][['🤗Checkpoints ']] (https://huggingface.co/yunyangx/efficient-track-anything/tree/main)

The **Efficient Track Anything Model(EfficientTAM)高效跟踪模型（EfficientTAM）** takes a vanilla lightweight ViT image encoder. An efficient memory cross-attention is proposed to further improve the efficiency. Our EfficientTAMs are trained on SA-1B (image) and SA-V (video) datasets. EfficientTAM achieves comparable performance with SAM 2 with improved efficiency. Our EfficientTAM can run **>10 frames per second   >每秒10帧** with reasonable video segmentation performance on **iPhone 15**. Try our demo with a family of EfficientTAMs at [[`🤗Gradio Demo`](https://10f00f01361a8328a4.gradio.live)].**高效跟踪任何模型(EfficientTAM)**需要一个香草轻量级ViT图像编码器。为了进一步提高效率，提出了一种高效的记忆交叉注意方法。我们的efficienttam是在SA-1B（图像）和SA-V（视频）数据集上训练的。EfficientTAM在提高效率的同时实现了与SAM 2相当的性能。我们的EfficientTAM可以在iPhone 15上以合理的视频分割性能运行每秒10帧。在[[‘🤗Gradio demo ’](https://10f00f01361a8328a4.gradio.live)])上尝试我们的演示。**高效跟踪任何模型(EfficientTAM)**需要一个香草轻量级ViT图像编码器。为了进一步提高效率，提出了一种高效的记忆交叉注意方法。我们的efficienttam是在SA-1B（图像）和SA-V（视频）数据集上训练的。EfficientTAM在提高效率的同时实现了与SAM 2相当的性能。我们的EfficientTAM可以在iPhone 15上以合理的视频分割性能运行每秒10帧。在[[‘🤗Gradio demo ’](https://10f00f01361a8328a4.gradio.live)])上尝试我们的演示。

![Efficient Track Anything design高效跟踪任何设计](figs/examples/overview.png无花果/ / overview.png例子)

## News   新闻
[Jan.5 2025] We add the support for running Efficient Track Anything on Macs with MPS backend. Check the example [2025年1月5日]我们添加了支持运行高效跟踪任何在mac与MPS后端。查看示例[app.py]（https://github.com/yformer/EfficientTAM/blob/main/app.py）。[app.py](https://github.com/yformer/EfficientTAM/blob/main/app.py).

[Jan.3 2025] We update the codebase of Efficient Track Anything, adpated from the latest [2025年1月3日]我们更新了Efficient Track Anything的代码库，采用了最新的[SAM2]（https://github.com/facebookresearch/sam2）代码库，提高了推理效率。请查看2024年12月11日最新的[SAM2]（https://github.com/facebookresearch/sam2）更新以了解详细信息。感谢SAM 2团队！[SAM2](https://github.com/facebookresearch/sam2) codebase with improved inference efficiency. Check the latest [SAM2](https://github.com/facebookresearch/sam2) update on Dec. 11 2024 for details. Thanks to SAM 2 team!

![Efficient Track Anything Speed Update高效跟踪任何速度更新](figs/examples/speed_vs_latency_update.png无花果/ / speed_vs_latency_update.png例子)

[Dec.22 2024] We release [2024年12月22日]我们发布[‘🤗高效跟踪任何检查点’]（https://huggingface.co/yunyangx/efficient-track-anything/tree/main）。[`🤗Efficient Track Anything Checkpoints🤗有效跟踪任何检查点`](https://huggingface.co/yunyangx/efficient-track-anything/tree/main).

[Dec.4 2024   2024年12] [2024年12月4日][‘🤗高效跟踪任何东西分割一切’]（https://5239f8e221db7ee8a0.gradio.live/）。感谢@SkalskiP！[`🤗Efficient Track Anything for segment everything🤗高效跟踪任何细分一切`](https://5239f8e221db7ee8a0.gradio.live/). Thanks to @SkalskiP!

[Dec.2 2024   按照2024年] We provide the preliminary version of Efficient Track Anything for demonstration.[2024年12月2日]我们提供了“高效跟踪任何东西”的初步版本进行演示。

## Online Demo & Examples   在线演示和示例
Online demo and examples can be found in the 在线演示和示例可以在[项目页面]（https://yformer.github.io/efficient-track-anything/）中找到。[project page   项目页面](https://yformer.github.io/efficient-track-anything/).

## EfficientTAM Video Segmentation Examples高效视频分割的例子
  |   |   |
:-------------------------:|:-------------------------:
SAM 2 [SAM2]（figs/examples/sam2_video_segmentation.png）| ![SAM2](figs/examples/sam2_video_segmentation.png无花果/ / sam2_video_segmentation.png例子)
EfficientTAM EfficientTAM （figs/examples/efficienttam_video_segmentation.png）|  ![EfficientTAM](figs/examples/efficienttam_video_segmentation.png无花果/ / efficienttam_video_segmentation.png例子)

## EfficientTAM Image Segmentation Examples高效tam图像分割示例
Input Image, SAM, EficientSAM, SAM 2, EfficientTAM
  |   |   |
:-------------------------:|:-------------------------:
Point-prompt [Point-prompt]（figs/examples/demo_img_point.png）| ![point-prompt](figs/examples/demo_img_point.png无花果/ / demo_img_point.png例子)[Point-prompt](figs/examples/demo_img_point.png)| ![Point-prompt]（figs/examples/demo_img_point.png）
Box-prompt [Box-prompt]（figs/examples/demo_img_box.png）|  ![box-prompt](figs/examples/demo_img_box.png无花果/ / demo_img_box.png例子)[Box-prompt](figs/examples/demo_img_box.png)| ![Box-prompt]（figs/examples/demo_img_box.png）
Segment everything 分割所有东西，b| ！(段一切)(无花果/ / demo_img_everything.png例子)|![segment everything   段的一切](figs/examples/demo_img_everything.png无花果/ / demo_img_everything.png例子)

## Model   模型
EfficientTAM checkpoints are available at the 在[拥抱空间]（https://huggingface.co/yunyangx/efficient-track-anything/tree/main）上可以找到有效的tam检查点。[Hugging Face Space   拥抱的脸部空间](https://huggingface.co/yunyangx/efficient-track-anything/tree/main).

## Getting Started   开始

### Installation   安装

```bash   ”“bash   ”“bash”“bash
git clone https://github.com/yformer/EfficientTAM.git
cd EfficientTAM
conda create -n efficient_track_anything python=3.12Conda create -n efficient_track_anything python=3.12conda create -n efficient_track_anything python=3.12 conda create -n efficient_track_anything python=3.12
conda activate efficient_track_anything激活efficient_track_anything激活efficient_track_anything
pip install -e .   PIP安装-e。PIP安装-e。PIP。
```
### Download Checkpoints   下载的检查点

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


### Building Efficient Track Anything###建立有效的跟踪
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

### Efficient Track Anything Notebook Example高效跟踪任何笔记本的例子
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
