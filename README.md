# BEP_Giel

hpo.py is the main file and should be run first. It includes parameters for the hyperparameter optimization search space and also calls either training_ae.py or training_rnn.py with a specific set of parameters.
evaluation - Copy.py generates evaluation statistics for all the trained models after running hpo.py.
hpo_statistics_evaluation.py generates summary statistics and plots from the evaluation mentioned above.

data_preprocessing.py, models.py, training_ae.py, training_rnn.py, and utils.py are included for importing functions or are called by one or multiple of the other files.
