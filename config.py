from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# general
cfg.seed = 42

# data
cfg.train_split = 0.7
cfg.val_split = 0.15


# training
cfg.lr = 0.000001
cfg.epochs = 20
