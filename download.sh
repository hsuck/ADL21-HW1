mkdir -p ./ckpt/intent/
mkdir -p ./ckpt/slot/

INTENT_PT_URL="https://www.dropbox.com/s/snsd33kquauilqs/best_model.pt?dl=1"
INTENT_PT_PATH="./ckpt/intent/best_model.pt"
wget -O ${INTENT_PT_PATH} ${INTENT_PT_URL}

SLOT_PT_URL="https://www.dropbox.com/s/5emw2ajpvh0au9v/best_model.pt?dl=1"
SLOT_PT_PATH="./ckpt/slot/best_model.pt"
wget -O ${SLOT_PT_PATH} ${SLOT_PT_URL}
