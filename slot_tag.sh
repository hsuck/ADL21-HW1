# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_slot.py --test_file "${1}" --pred_file "${2}" --dropout 0.3 --num_layers 2 --hidden_size 512 --ckpt_path ckpt/slot/CNN_GRU_best_model.pt --model CNN_GRU --num_cnn 2
