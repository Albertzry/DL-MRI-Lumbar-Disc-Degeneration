#preprocessing
nnFormer_convert_decathlon_task -i ./DATASET/nnFormer_raw/nnFormer_raw_data/Task01_disc
nnFormer_plan_and_preprocess -t 1

#train
bash train_inference.sh -c 0 -n nnformer_disc -t 1 

#predict
python3 -m nnformer.inference.predict_simple \
  -i /root/DL-MRI-Lumbar-Disc-Degeneration/DATASET/nnFormer_raw/nnFormer_raw_data/Task001_disc/imagesTs \
  -o /root/DL-MRI-Lumbar-Disc-Degeneration/inferTs/nnformer_pred \
  -t Task001_disc \
  -m 3d_fullres \
  -f 0 \
  -chk model_best \
  -tr nnFormerTrainerV2

#test
ln -s /root/DL-MRI-Lumbar-Disc-Degeneration/DATASET/nnFormer_raw/nnFormer_raw_data/Task01_disc/labelsTs ./labelsTs
python3 nnformer/interence_disc.py nnformer_pred --threshold 0.5