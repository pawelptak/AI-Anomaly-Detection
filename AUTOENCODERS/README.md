# AUTOENCODERS

This application contains various autoencoders with examples of usage and results analysis on the actual data.
It includes Dense Autoencoders and LSTM autoencoders

## The Application is splitted into 6 moduls:

1. Data - it contains actual data (if is not too big) or intructions/link how to download the actual data
2. DataPreparing - it contains classes for Loading and Precprocessing the Data
3. Model - it contains models definitions
4. Execution - it contains code responsible for executing the Models on actual Data
5. Classification - it contains code for performing the classification on the trained Model
6. Analisis - it contains code for performing the analysis of the results based on the model output.

## Example Usage

All the files with names starting with AE\_ are examples of usage.
We can find there various combinations of data and applied model as well as different preprocessing strategies.

The example usage flow is as follow:

1. Load the specified data
   - different datasets may be choosen
2. Preprocess train and test data
   - preprocessing should be adjusted to the data
   - different datasets may require different different preprocessing operations
   - the results of preprocessing may be single lines (vectors) or windows
   - to extend the exesting solution - just add a valid preprocess to the DataPreparing module
3. Create the specified model
   - different models may be used
   - to create your own model just add it to the Model modul
4. Fit the model and predict the results
   - this operation should looks the same in all cases, not matter what data and what model were used
5. Classify the results (as anomally or not)
   - this should not depends on model or data, but may relay on preprocessing output - if it is in single lines format or windows
   - computes the threshold
6. Analysis
   - performing the analysis on the results
   - create and save confusion matrix and results diagrams

For an example with comments please take a look into AE_1.py file.

AE_1.py file represents an example of Dense model working with KDD data (in single lines format).
AE_11.py file represents an example of LSTM model working with KDD data (in windows format).

Other files are examples of combination of different datasets, data format and models.

## Extensions

The following extension options are available

1. New Data Set
   In order to work with new dataset creating new Loader and Preprocessor is required.

   - The loader should implement two methods - to load train and test data. Please take a look on an example KDDLoader.py designed to work with KDD data.
   - The preprocessor should implement two methods - one for preprocessing the train data and the other one for preprocessing the test or any additional data we would like to predict. Please take a look on KDDPreprocessor.py which includes additional comments explaing what exactly it's doing. It offers representing data in two differetn formats: as single lines or as windows. CAUTION: Not every model is adjusted to work with every data type (e.g. LSTM model only works with data in windows format)

2. New Model
   In order to introduce new model just create a new model class in Model module. Inherit from AutoEncoderModelBase class and implement the build method.
   OOTB two main models are availbe:

   - Dense - it is build mostly based on Dense Layers (please take a look into AutoEncoderModelDense.py)
   - LSTM - it is build mostly based on LSTM Layers (please take a look into AutoEncoderModelLSTM.py)

   The models offer some configuration parameters like number of layers or number of units.
