DBS = ["FB15k-237"] #, "NELL-995-h100", "FB15k-237", "WN18RR"
# MODELS = ["Pykeen_RotatE", "Pykeen_MuRE" ,"Keci"]
 
MODELS = ["Pykeen_TransE"]
#'Pykeen_TransE', 'Pykeen_TransH', "DistMult", "ComplEx", "DeCaL"


BATCH_SIZE = "1024"
LEARNING_RATE = "1e-3"

NUM_EPOCHS = "100"
EMB_DIM = "32"
LOSS_FN = "BCELoss"
SCORING_TECH = "KvsAll"
OPTIM = "Adam"

#for bayesian optimization, use train_val_test
EVAL_MODEL_TRAIN_VAL_TEST = "train_val_test"

#for actual experiments, use test
EVAL_MODEL_TEST = "test"