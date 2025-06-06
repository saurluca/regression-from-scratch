from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# general
cfg.seed = 42

# data
cfg.train_split = 0.7
cfg.val_split = 0.15
# test_split = 1.0 - 0.7 - 0.15 = 0.15

# training
cfg.lr_manual = 0.0001  # learning rate
cfg.lr_full_batch = 0.0001  # learning rate 
cfg.epochs_manual = 20  
cfg.epochs_full_batch = 2000  

# Logistic regression
cfg.lr_logistic_manual = 0.05
cfg.lr_logistic_full_batch = 0.005
cfg.epochs_logistic_manual = 60
cfg.epochs_logistic_full_batch = 1500
cfg.results_dir_logistic = "results_logistic"
