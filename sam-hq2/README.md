# HQ-SAM 2: Segment Anything in High Quality for Images and Videos    


We propose **HQ-SAM2** to upgrade SAM2 to **higher quality** by extending our training strategy in [HQ-SAM](https://arxiv.org/abs/2306.01567). 

## Latest updates

**2024/11/17 -- HQ-SAM 2 is released**

- A new suite of improved model checkpoints (denoted as **HQ-SAM 2**, **beta-version**) are released. See [Model Description](#model-description) for details.

![HQ-SAM2 results comparison](assets/hq-sam2-results.png?raw=true)

## Installation

HQ-SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM 2 on a GPU machine using:

```bash
git clone https://github.com/SysCV/sam-hq.git
conda create -n sam_hq2 python=3.10 -y
conda activate sam_hq2
cd sam-hq/sam-hq2
pip install -e .
```
If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

To use the HQ-SAM 2 predictor and run the example notebooks, `jupyter` and `matplotlib` are required and can be installed by:

```bash
pip install -e ".[notebooks]"
```

Note:
1. It's recommended to create a new Python environment via [Anaconda](https://www.anaconda.com/) for this installation and install PyTorch 2.3.1 (or higher) via `pip` following https://pytorch.org/. If you have a PyTorch version lower than 2.3.1 in your current environment, the installation command above will try to upgrade it to the latest PyTorch version using `pip`.
2. The step above requires compiling a custom CUDA kernel with the `nvcc` compiler. If it isn't already available on your machine, please install the [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.
3. If you see a message like `Failed to build the SAM 2 CUDA extension` during installation, you can ignore it and still use SAM 2 (some post-processing functionality may be limited, but it doesn't affect the results in most cases).

Please see [`INSTALL.md`](./INSTALL.md) for FAQs on potential issues and solutions.

## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

or individually from:

<!-- - [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) -->
- [sam2.1_hq_hiera_large.pt](https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt?download=true)

(note that these are the improved checkpoints denoted as SAM 2.1; see [Model Description](#model-description) for details.)

Then HQ-SAM 2 can be used in a few lines as follows for image and video prediction.

### Image prediction

HQ-SAM 2 has all the capabilities of [HQ-SAM](https://github.com/SysCV/sam-hq) on static images, and we provide image prediction APIs that closely resemble SAM for image use cases. The `SAM2ImagePredictor` class has an easy interface for image prompting.

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# Baseline SAM2.1
# checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Ours HQ-SAM 2
checkpoint = "./checkpoints/sam2.1_hq_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>, multimask_output=False)
```

Please refer to the examples in [python demo/demo_hqsam2.py](./demo/demo_hqsam2.py) for details on how to add click or box prompts.

Please refer to the examples in [image_predictor_example.ipynb](./notebooks/image_predictor_example.ipynb) for static image use cases.

### Video prediction

For promptable segmentation and tracking in videos, we provide a video predictor with APIs for example to add prompts and propagate masklets throughout a video. SAM 2 supports video inference on multiple objects and uses an inference state to keep track of the interactions in each video.

```python
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2_hq_video_predictor
# Baseline SAM2.1
# checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Ours HQ-SAM 2
checkpoint = "./checkpoints/sam2.1_hq_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
predictor = build_sam2_hq_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```


Please refer to the examples in [video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb) for static image use cases.


## Model Description

### HQ-SAM 2 checkpoints

The table below shows the **zero-shot** image segmentation performance of SAM2.1 and HQ-SAM 2 on **COCO (AP)** using same bounding box detector from Focal-net DINO. The FPS speed of SAM2.1 and HQ-SAM 2 is on par.
|      **Model**       | **Size (M)** | **Single Mode (AP)** | **Multi-Mode (AP)** |
| :------------------: | :----------: | :-----------------: | :----------------: |
|   sam2.1_hiera_large <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt))   |    224.4     |        50.0         |        48.3       |
|   sam2.1_hq_hiera_large <br /> ([config](sam2/configs/sam2.1/sam2.1_hq_hiera_l.yaml), [checkpoint](https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt?download=true))   |    224.7     |        **50.9**         |        **50.4**       |

The table below shows the **zero-shot** image segmentation AP performance of Grounded-SAM 2 and Grounded-HQ-SAM 2 on [**Seginw (Segmentation in the Wild)** dataset](https://github.com/SysCV/sam-hq/tree/main/seginw).


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model Name</th>
<th valign="bottom">SAM</th>
<th valign="bottom">GroundingDINO</th>
<th valign="bottom">Mean AP</th>
<th valign="bottom">Airplane-Parts</th>
<th valign="bottom">Bottles</th>
<th valign="bottom">Brain-Tumor</th>
<th valign="bottom">Chicken</th>
<th valign="bottom">Cows</th>
<th valign="bottom">Electric-Shaver</th>
<th valign="bottom">Elephants</th>
<th valign="bottom">Fruits</th>
<th valign="bottom">Garbage</th>
<th valign="bottom">Ginger-Garlic</th>
<th valign="bottom">Hand-Metal</th>
<th valign="bottom">Hand</th>
<th valign="bottom">House-Parts</th>
<th valign="bottom">HouseHold-Items</th>
<th valign="bottom">Nutterfly-Squireel</th>
<th valign="bottom">Phones</th>
<th valign="bottom">Poles</th>
<th valign="bottom">Puppies</th>
<th valign="bottom">Rail</th>
<th valign="bottom">Salmon-Fillet</th>
<th valign="bottom">Strawberry</th>
<th valign="bottom">Tablets</th>
<th valign="bottom">Toolkits</th>
<th valign="bottom">Trash</th>
<th valign="bottom">Watermelon</th>
<!-- TABLE BODY -->
 <tr><td align="left">Grounded SAM2</td>
<td align="center">vit-l</td>
<td align="center">swin-b</td>
<td align="center">49.5</td>
<td align="center">38.3</td>
<td align="center">67.1</td>
<td align="center">12.1</td>
<td align="center">80.7</td>
<td align="center">52.8</td>
<td align="center">72.0</td>
<td align="center">78.2</td>
<td align="center">83.3</td>
<td align="center">26.0</td>
<td align="center">45.7</td>
<td align="center">73.7</td>
<td align="center">77.6</td>
<td align="center">8.6</td>
<td align="center">60.1</td>
<td align="center">84.1</td>
<td align="center">34.6</td>
<td align="center">28.8</td>
<td align="center">48.9</td>
<td align="center">14.3</td>
<td align="center">24.2</td>
<td align="center">83.7</td>
<td align="center">29.1</td>
<td align="center">20.1</td>
<td align="center">28.4</td>
<td align="center">66.0</td>
</tr>

 <tr><td align="left">Grounded HQ-SAM2</td>
<td align="center">vit-l</td>
<td align="center">swin-b</td>
<td align="center"><b>50.0</b></td>
<td align="center">38.6</td>
<td align="center">66.8</td>
<td align="center">12.0</td>
<td align="center">81.0</td>
<td align="center">52.8</td>
<td align="center">71.9</td>
<td align="center">77.2</td>
<td align="center">83.3</td>
<td align="center">26.1</td>
<td align="center">45.5</td>
<td align="center">74.8</td>
<td align="center">79.0</td>
<td align="center">8.6</td>
<td align="center">60.1</td>
<td align="center">84.7</td>
<td align="center">34.3</td>
<td align="center">25.5</td>
<td align="center">48.9</td>
<td align="center">14.1</td>
<td align="center">34.1</td>
<td align="center">85.7</td>
<td align="center">29.2</td>
<td align="center">21.5</td>
<td align="center">28.9</td>
<td align="center">66.6</td>

</tr>
</tbody></table>


The table below shows the **zero-shot** video object segmentation performance of SAM2.1 and HQ-SAM 2.
|      **Model**       | **Size (M)** | **DAVIS val (J&F)** | **MOSE(J&F)** |
| :------------------: | :----------: |:----------------: | :---------------: |
|   sam2.1_hiera_large <br /> ([config](sam2/configs/sam2.1/sam2.1_hiera_l.yaml), [checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt))   |    224.4     |        89.8        |       74.6        |
|   sam2.1_hq_hiera_large <br /> ([config](sam2/configs/sam2.1/sam2.1_hq_hiera_l.yaml), [checkpoint](https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt?download=true))   |    224.7     |        **91.0**        |       **74.7**        |



## License

The HQ-SAM 2, SAM 2 model checkpoints, SAM 2 demo code (front-end and back-end), and SAM 2 training code are licensed under [Apache 2.0](./LICENSE), however the [Inter Font](https://github.com/rsms/inter?tab=OFL-1.1-1-ov-file) and [Noto Color Emoji](https://github.com/googlefonts/noto-emoji) used in the SAM 2 demo code are made available under the [SIL Open Font License, version 1.1](https://openfontlicense.org/open-font-license-official-text/).

## Citing HQ-SAM 2
If you find HQ-SAM2 useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{sam_hq,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    booktitle={NeurIPS},
    year={2023}
}  
```
