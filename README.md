# MICCAI2024

first:
  run `process.ipynb` => process the data 

second:
  run `train_bin.sh` to get the weight of binary segmentation

three:
  run  `add_roi_mask.ipynb` to add binary mask

last:
  run `train_multitask.sh` to get weight of instance segmentation
