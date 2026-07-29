# mlops-weather-forecast
* This is an end-to-end MLOps pipeline project   
* This projects gets daily weather data from the Historical Weather API  
* Builds GBM forecast  
* Deploys the model with FastAPI and Docker  
* Reruns the model building pipeline daily using Github Actions  
* Provides a model/data drift report using Evidently  
* Logs all model runs in MLflow (Local)  

# next steps
* Deploy model to a container in the cloud  
* Have a cloud instance of MLflow, as Github actions spins up a new instance every day  
* Improve the actual model performance as the goal of v1 was to simply have a complete deployed solution instead of a highly accurate notebook  
