# Bengali-Sentiment-Analysis-on-MT-Dataset

Train:
To train model please execute the script according to classification algorithm name. For example to train model using hybrid LSTMCNN please run LSTMCNNV4.py
Aditionally, also update the location, tokenizer_model_name parameter within the script to specify which dataset to use for training.

Testing:
To test the models performance use according to the classification algorithm that was used to train the model. For example if model was trained using hybrid LSTMCNNV4.py, please execute Eval_LLSTMCNNV4.py.
Aditionally, need to update tokenizer_model_name, location_of_testdata, output_file_name before executing. Here tokenizer_model_name should be same as was specified during training, location_of_testdata is the location of test dataset and output_file_name should be same as location_of_testdata.

Pretrained Model:
The pretrained models are available for testing