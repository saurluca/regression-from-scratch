
from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# general
cfg.seed = 42

# data
cfg.train_split = 0.7
cfg.val_split = 0.15


# training
cfg.learning_rate = 0.01
cfg.batch_size = 32
cfg.epochs = 10

# model
cfg.hidden_units = 10
cfg.dropout = 0.2

