export CUDA_VISIBLE_DEVICES=-1
rm output/__result__/*WGAN3
python3 launcher.py bgmwgan3 --dataset chain --repeat 1 &
# python3 launcher.py bgmwgan2 --dataset chain --repeat 1 &
# python3 launcher.py bgmgan4 --dataset chain --repeat 1 &
# python3 launcher.py bgmgan5 --dataset chain --repeat 1 &
