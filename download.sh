mkdir -p ./ckpt/intent/
mkdir -p ./ckpt/slot/

INTENT_PT_URL="https://www.dropbox.com/s/4lrgew5ruicywx6/best_model.pt?dl=1"
INTENT_PT_PATH="./ckpt/intent/best_model.pt"
wget -O ${INTENT_PT_PATH} ${INTENT_PT_URL}

SLOT_PT_URL="https://www.dropbox.com/s/5v0rnyowqtmw0rq/best_model.pt?dl=1"
SLOT_PT_PATH="./ckpt/slot/best_model.pt"
wget -O ${SLOT_PT_PATH} ${SLOT_PT_URL}
