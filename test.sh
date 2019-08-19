# CUDA_VISIBLE_DEVICES=0 python test.py --batch-size 5 --start-epoch 5 --margin 0.5 \
# --dataset coco --num-train-triplets 100000 --num-valid-triplets 5000 --num-classes 80

CUDA_VISIBLE_DEVICES=0 python test.py --batch-size 15 --start-epoch 64 --margin 0.5 \
--dataset pascal_voc --num-test-triplets 3000 --num-valid-triplets 3000 --num-classes 20 \
--learning-rate 0.0001