PYTHON=/usr/bin/python27
PYTHON=python

EPOCHS=15
NAME=experiment

LOGFOLDER=log
DATASET=mnist
MODEL=rbm
ALG=adam

LR=.1
B1=0.9
B2=0.999
SUPERBATCH=1024
NB=20

# ----------------------------------------------------------------------------

train:
	$(PYTHON) run.py train \
	  --dataset $(DATASET) \
	  --model $(MODEL) \
	  -e $(EPOCHS) \
	  --logname $(LOGFOLDER)/$(DATASET).$(MODEL).$(ALG).$(LR).$(NB).$(NAME) \
	  --plotname $(LOGFOLDER)/$(DATASET)_$(MODEL)_$(ALG)_$(LR)_$(NB)_$(NAME).png \
	  --alg $(ALG) \
	  --lr $(LR) \
	  --b1 $(B1) \
	  --b2 $(B2) \
	  --n_superbatch $(SUPERBATCH) \
	  --n_batch $(NB)
