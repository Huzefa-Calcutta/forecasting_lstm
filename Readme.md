Model Training

1. cd to the root folder
2. Install all package dependencies as    
    `pip install -r requirements.txt`
3. specify the location training data file location and ouptut folder where you want to save the model in model.config file   
4. execute python train.py command   
    `python train.py model.cfg`
    
Model Inference
1. specify the location future data file, historical data file location and output folder where you want to save the results in model.config file   
2. execute python inference.py command 
    `python inference.py inference.cfg`

   