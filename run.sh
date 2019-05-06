export CUDA_VISIBLE_DEVICES=0
python3 launcher.py bgmwgan3 --datasets grid ring fc general chain tree --repeat 3 &
python3 launcher.py bgmwgan3 --datasets mnist12 --repeat 3 &
python3 launcher.py bgmwgan3 --datasets mnist28 --repeat 3 &
export CUDA_VISIBLE_DEVICES=1
python3 launcher.py bgmwgan3 --datasets news --repeat 3 &
python3 launcher.py bgmwgan3 --datasets adult --repeat 3 &
python3 launcher.py bgmwgan3 --datasets credit --repeat 3 &
