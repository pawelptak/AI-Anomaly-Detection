from tensorflow.python.keras.backend import constant
from Classification.AutoencoderResultsClassificator import AutoencoderResultsClassificator
from DataPreparing.CICIDSLoaderWindows import CICIDSLoaderWindows
from DataPreparing.CICIDSPreprocessorWindows import CICIDSPreprocessorWindows
from Execution.AutoencoderModelExecutor import AutoencoderModelExecutor
from Model.AutoEncoderModelDense import AutoEncoderModelDense_1, AutoEncoderModelDense_2
import numpy as np
import matplotlib.pyplot as plt
from Model.AutoEncoderModelLSTM import AutoEncoderModelLSTM
from Classification.AutoencoderWindowsResultsClassificator import AutoencoderWindowsResultsClassificator
from Analisis.AutoencoderResultsAnlyzer import AutoencoderResultsAnlyzer


ATTACK = "DDoS"
NORMAL = "BENIGN"
WINDOW_SIZE = 20
STRIDE = 10
UNITS = [80, 50, 20]


loader = CICIDSLoaderWindows(ATTACK)
preprocessor = CICIDSPreprocessorWindows(WINDOW_SIZE, STRIDE)

df = loader.load_train_data()
df_test = loader.load_test_data()

df = preprocessor.preprocess_train_data(df, NORMAL)
(windows, y_label) = preprocessor.preprocess_test_data(df_test, NORMAL)

y_label = np.array(y_label)

model = AutoEncoderModelLSTM(df.shape[2], WINDOW_SIZE, UNITS)
model_executor = AutoencoderModelExecutor(model, epochs=3)
classificator = AutoencoderWindowsResultsClassificator()
analyzer = AutoencoderResultsAnlyzer()

model_executor.fit(df)

x_predicted = model_executor.predict(windows)

classificator.feed(windows, x_predicted, y_label)

error = classificator.calculate_reconstruction_error_windows()

max_f1_score, best_threshold = classificator.calculate_best_threshold()
analyzer.feed(error, y_label, best_threshold,
              max_f1_score, f'DDoS/LSTM_M_{WINDOW_SIZE}', f'DDoS, window size = {WINDOW_SIZE}')

analyzer.plot_results()
analyzer.plot_confusion_matrix()
