from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# general
cfg.seed = 42

# data
cfg.train_split = 0.7
cfg.val_split = 0.15


# training
cfg.lr_manual = 0.0001
cfg.lr_full_batch = 500
cfg.epochs_manual = 20
cfg.epochs_full_batch = 1000
