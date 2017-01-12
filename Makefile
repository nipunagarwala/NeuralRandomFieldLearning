PYTHON=/usr/bin/python27
PYTHON=python

EPOCHS=200
NAME=experiment

LOGFOLDER=log
DATASET=mnist
MODEL=adgm_gsm
ALG=adam

LR=1e-3
B1=0.9
B2=0.999
SUPERBATCH=1024
NB=256

# ----------------------------------------------------------------------------

train:
	$(PYTHON) run.py train \
	  --dataset $(DATASET) \
	  --model $(MODEL) \
	  -e $(EPOCHS) \
	  -l $(LOGFOLDER)/$(DATASET).$(MODEL).$(ALG).$(LR).$(NB).$(NAME) \
	  --alg $(ALG) \
	  --lr $(LR) \
	  --b1 $(B1) \
	  --b2 $(B2) \
	  --n_superbatch $(SUPERBATCH) \
	  --n_batch $(NB)
