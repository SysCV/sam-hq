#!/bin/bash  
  
for file in ./data/seginw/*;  
do  
echo $file is data path \! ;  

python test_ap_on_seginw.py -c GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py -p pretrained_checkpoint/groundingdino_swinb_cogcoor.pth --anno_path $file/valid/_annotations_min1cat.coco.json --image_dir $file/valid/ --use_sam_hq
done