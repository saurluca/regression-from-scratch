from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# general
cfg.seed = 42

# data
cfg.train_split = 0.7
cfg.val_split = 0.15
# test_split = 1.0 - 0.7 - 0.15 = 0.15

# training
cfg.lr_manual = 0.0001          # learning rate - nicht vektorisiert (?)
cfg.lr_full_batch = 500         # learning rate - vektorisiert (ist das nicht sehr hoch?)
cfg.epochs_manual = 20          # durchläufe - nicht vektorisiert
cfg.epochs_full_batch = 1000    # durchläufe - vektorisiert

# Logistic regression
cfg.lr_logistic_manual = 0.01
cfg.lr_logistic_full_batch = 0.005 # Beispielwerte
cfg.epochs_logistic_manual = 50   # Beispielwerte
cfg.epochs_logistic_full_batch = 1500 # Beispielwerte
cfg.results_dir_logistic = 'results_logistic'