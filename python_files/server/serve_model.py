"""
The main server model constructor and evaluator.

This reads data, model choice, and seal context obj from the client; calls the 
model config file (which contains everything needed to set up model); encoded params
to model, sets up model (with params encoded); then evaluates the model; and sends 
the encrypted predictions back to the client.

"""

#TODO

#Injest encrypted data, context (security parameter is a property)

#Pull model parameters file and encode params

#Set up seal_model with parameters. 

#Evaluate model on encrypted data

#Send encrypted predictions.