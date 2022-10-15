# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_intent.py --test_file "${1}" --ckpt_path ckpt/intent/GRU_best_model.pt --pred_file "${2}" --dropout 0.2 --num_layers 2 --hidden_size 512 --model GRU
