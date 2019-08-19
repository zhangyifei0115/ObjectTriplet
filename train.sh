# CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 5 --margin 0.5 \
# --dataset coco --num-train-triplets 10000 --num-valid-triplets 3000 --num-classes 80

CUDA_VISIBLE_DEVICES=0 python train.py --batch-size 15 --margin 0.5 \
--dataset pascal_voc --num-train-triplets 50000 --num-valid-triplets 5000 --num-classes 20 \
--learning-rate 0.000001 --num-epochs 500
