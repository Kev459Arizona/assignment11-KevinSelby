# assignment11-KevinSelby
In this assignment, we had a model deployed on our droplet to be used for inference.
# Part 1: Requirements
Requirements.txt was used to install all of our app dependecies inclusing flask for inference, mlflow, and scikit-learn for the model
# Part 2: APP
In app.py, we actaully train the model with a random forest.
# Part 3: Docker
For part three, we create a dockerfile that allows us to containerize our model with all of the necessary code. The docker-compose file allows our container to be built correctly.
# Part 4: Droplet
After all of these files have been created, we create a droplet to actually run our model. We use SSH encryption and SCP in order to move the files over to the droplet.
# Part 5: Inference
Finally, we build the image, start the container, and send input data via flask to our server for inference
