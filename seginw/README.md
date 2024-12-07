# Results comparison on the [Segmentation in the Wild benchmark](https://eval.ai/web/challenges/challenge-page/1931/overview?ref=blog.roboflow.com)

> [**Segment Anything in High Quality**](https://arxiv.org/abs/2306.01567)           
> Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu \
> ETH Zurich & HKUST 

We organize the `seginw` folder as follows.
```
seginw
|____data
|____pretrained_checkpoint
|____GroundingDINO
|____segment_anything
|____test_ap_on_seginw.py
|____test_seginw.sh
|____test_seginw_hq.sh
|____logs
```

## 1. Environment setup (only required for SegInW)
```
cd seginw
python -m pip install -e GroundingDINO
```

## 2. Evaluation Data Preparation

Seginw (Segmentation in the Wild) dataset can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/SegInW/resolve/main/seginw.zip)
```
cd data
wget https://huggingface.co/sam-hq-team/SegInW/resolve/main/seginw.zip
unzip seginw.zip
```

### Expected dataset structure for [SegInW](https://eval.ai/web/challenges/challenge-page/1931/overview?ref=blog.roboflow.com)

```
data
|____seginw
| |____Airplane-Parts
| |____Bottles
| |____Brain-Tumor
| |____Chicken
| |____Cows
| |____Electric-Shaver
| |____Elephants
| |____Fruits
| |____Garbage
| |____Ginger-Garlic
| |____Hand
| |____Hand-Metal
| |____House-Parts
| |____HouseHold-Items
| |____Nutterfly-Squireel
| |____Phones
| |____Poles
| |____Puppies
| |____Rail
| |____Salmon-Fillet
| |____Strawberry
| |____Tablets
| |____Toolkits
| |____Trash
| |____Watermelon
```

## 3. Pretrained Checkpoint
Init checkpoint can be downloaded by

```
cd pretrained_checkpoint
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
wget https://huggingface.co/sam-hq-team/sam-hq-training/resolve/main/pretrained_checkpoint/sam_vit_h_4b8939.pth
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth
```

### Expected checkpoint

```
pretrained_checkpoint
|____groundingdino_swinb_cogcoor.pth
|____sam_hq_vit_h.pth
|____sam_vit_h_4b8939.pth
```


## 4. Evaluation
To evaluate on 25 seginw datasets

```
# baseline Grounded SAM
bash test_seginw.sh

# Grounded HQ-SAM
bash test_seginw_hq.sh
```

To evaluate sam2 and [sam-hq2](../sam-hq2/README.md)
```
# baseline Grounded SAM
bash test_seginw_sam2.sh

# Grounded HQ-SAM
bash test_seginw_sam_hq2.sh
```

### Example evaluation script on a single dataset
```
python test_ap_on_seginw.py -c GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py -p pretrained_checkpoint/groundingdino_swinb_cogcoor.pth --anno_path data/seginw/Airplane-Parts/valid/_annotations_min1cat.coco.json --image_dir data/seginw/Airplane-Parts/valid/ --use_sam_hq --save_json

```



## 5. Detailed Results on SegInW

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
<!-- ROW: maskformer2_R50_bs16_50ep -->
 <tr><td align="left">Grounded SAM</td>
<td align="center">vit-h</td>
<td align="center">swin-b</td>
<td align="center">48.7</td>
<td align="center">37.2</td>
<td align="center">65.4</td>
<td align="center">11.9</td>
<td align="center">84.5</td>
<td align="center">47.5</td>
<td align="center">71.7</td>
<td align="center">77.9</td>
<td align="center">82.3</td>
<td align="center">24.0</td>
<td align="center">45.8</td>
<td align="center">81.2</td>
<td align="center">70.0</td>
<td align="center">8.4</td>
<td align="center">60.1</td>
<td align="center">71.3</td>
<td align="center">35.4</td>
<td align="center">23.3</td>
<td align="center">50.1</td>
<td align="center">8.7</td>
<td align="center">32.9</td>
<td align="center">83.5</td>
<td align="center">29.8</td>
<td align="center">20.8</td>
<td align="center">30.0</td>
<td align="center">64.2</td>
</tr>
<!-- ROW: maskformer2_R101_bs16_50ep -->
 <tr><td align="left">Grounded HQ-SAM</td>
<td align="center">vit-h</td>
<td align="center">swin-b</td>
<td align="center"><b>49.6</b></td>
<td align="center">37.6</td>
<td align="center">66.3</td>
<td align="center">12.0</td>
<td align="center">84.5</td>
<td align="center">47.8</td>
<td align="center">72.1</td>
<td align="center">77.5</td>
<td align="center">82.3</td>
<td align="center">25.0</td>
<td align="center">45.6</td>
<td align="center">81.2</td>
<td align="center">74.8</td>
<td align="center">8.5</td>
<td align="center">60.1</td>
<td align="center">77.1</td>
<td align="center">35.3</td>
<td align="center">20.1</td>
<td align="center">50.1</td>
<td align="center">7.7</td>
<td align="center">42.2</td>
<td align="center">85.6</td>
<td align="center">29.7</td>
<td align="center">21.8</td>
<td align="center">30.0</td>
<td align="center">65.6</td>

</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_50ep -->
</tbody></table>

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
