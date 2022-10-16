mkdir -p ./ckpt/intent/
mkdir -p ./ckpt/slot/

INTENT_PT_URL="https://www.dropbox.com/s/nrbc5h5tky1pl2p/GRU_best_model.pt?dl=1"
INTENT_PT_PATH="./ckpt/intent/GRU_best_model.pt"
wget -O ${INTENT_PT_PATH} ${INTENT_PT_URL}

SLOT_PT_URL="https://www.dropbox.com/s/zx5u177k8ji65q0/CNN_GRU_best_model.pt?dl=1"
SLOT_PT_PATH="./ckpt/slot/CNN_GRU_best_model.pt"
wget -O ${SLOT_PT_PATH} ${SLOT_PT_URL}
