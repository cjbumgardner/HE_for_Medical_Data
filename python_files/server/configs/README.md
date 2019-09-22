This directory is for storing all of the configuration files for the models including their training config as well as evaluating config. It is assumed that each model's parameters have aleady been stored with the appropriate truncated precision. Ideally they are trained with those restricted truncations in mind and optimized to balance accruacy of the model while needing a small number of critical digits. 

The evaluating config file should be stored as "eval_(MODELNAME)_config" and have all of the following:
    - int_prec: numeric precision of the whole part of parameters for the model.
    - frac_prec: decimal precision for model parameters. 
    - path_to_model_params: relative path to model parameters
    - encode_base: encoding base to express rational numbers in

NOTE: We don't store encoded parameters because we let the client determine the security level/ speed trade-off. In the future a file could be stored that caches the encoded parameters if the client wishes to just stick with a security level. 

The train config should be stored as "train_(MODELNAME)_config" and have all of the following: 
    - relevant training hyperparameters (e.g. learning rate)
    - path to training data (which will have train, test, eval)
    