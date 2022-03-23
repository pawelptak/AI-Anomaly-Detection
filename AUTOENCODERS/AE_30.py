from Classification.AutoencoderResultsClassificator import AutoencoderResultsClassificator
from DataPreparing.EVENTSEQUENCELoader import EVENTSEQUENCELoader
from DataPreparing.EVENTSEQUENCEPreprocessor import EVENTSEQUENCEPreprocessor
from Execution.AutoencoderModelExecutor import AutoencoderModelExecutor
from Analisis.AutoencoderResultsAnlyzer import AutoencoderResultsAnlyzer
import numpy as np
import matplotlib.pyplot as plt
from Model.AutoEncoderModelLSTM import AutoEncoderModelLSTM
from Classification.AutoencoderWindowsResultsClassificator import AutoencoderWindowsResultsClassificator

UNITS = [80, 50, 20]

loader = EVENTSEQUENCELoader()
preprocessor = EVENTSEQUENCEPreprocessor()

df = loader.load_train_data()
df_test = loader.load_test_data()

train_windows = preprocessor.preprocess_train_data(df)
(test_windows, y_label) = preprocessor.preprocess_test_data(df_test)

model = AutoEncoderModelLSTM(
    train_windows.shape[2], train_windows.shape[1], UNITS)
model_executor = AutoencoderModelExecutor(model, epochs=15)
classificator = AutoencoderWindowsResultsClassificator()
analyzer = AutoencoderResultsAnlyzer()

model_executor.fit(train_windows)

x_predicted = model_executor.predict(test_windows)

classificator.feed(test_windows, x_predicted, y_label)

error = classificator.calculate_reconstruction_error_windows()

max_f1_score, best_threshold = classificator.calculate_best_threshold()
analyzer.feed(error, y_label, best_threshold,
              max_f1_score, f'EVENTSEQUENCE/LSTM_chunks', f'EVENTSEQUENCE')

analyzer.plot_results()
analyzer.plot_confusion_matrix()
