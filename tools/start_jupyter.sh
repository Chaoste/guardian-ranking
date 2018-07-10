export CUDA_VISIBLE_DEVICES=3
jupyter notebook --port 8888 --ip 0.0.0.0 > notebok.log &
tail -f notebook.log
