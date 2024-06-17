# Vit-Sea-ice-and-lead-classifier 
This repository contains a complete workflow for classification of sea-ice and lead specifically for OLCI imagery (surely can be adapted to other images at ease). 
You can follow the steps to train a model yourself or use the model that is pre-trained on around 80 OLCI images (about 10,000 data points) 
instead of training your own models.
This repository is quite straight forward and easy to follow for two purposes:
* Learn to train your own model from scartch 
* Implement the model straightaway

## Follow these steps if you want to start from scartch
* [Data pre-processing](Data_Pre-processing)
* [Model Training](Model_Training)
* [Implementation](Implementation_on_a_full_image)

Please go to each folder one by one and explanation has been given (either in the introduction of each folder or in the specific python file)

## Follow these steps if you want use the model directly

You need to download all the files in [Pre-trained model](Pre_trained_model)

You can directly load the model in [Implementation](Implementation_on_a_full_image/Implementation_On_Full_Image.ipynb)
using 
``` 
model = keras.models.load_model('Path') 
```
And follow all the steps in the code file [Implementation](Implementation_on_a_full_image/Implementation_On_Full_Image.ipynb)
