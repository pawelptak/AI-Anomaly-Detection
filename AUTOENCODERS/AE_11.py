from tensorflow.python.ops.gen_math_ops import erf_eager_fallback
from Classification.AutoencoderWindowsResultsClassificator import AutoencoderWindowsResultsClassificator
from DataPreparing.KDDLoader import KDDLoader
from DataPreparing.KDDPreprocessor import KDDPreprocessor
from Execution.AutoencoderModelExecutor import AutoencoderModelExecutor
from Model.AutoEncoderModelLSTM import AutoEncoderModelLSTM
import numpy as np
import matplotlib.pyplot as plt
from Analisis.AutoencoderResultsAnlyzer import AutoencoderResultsAnlyzer


SERVICE = "http"
TRAIN_LABEL = "normal"
TEST_LABEL = "neptune"  # back / neptune
#TRAIN_LABEL = 0
#TEST_LABEL = 1

UNITS = [80, 50, 20]
WINDOW_SIZE = 20
STRIDE = 10

loader = KDDLoader(SERVICE)
preprocessor = KDDPreprocessor(
    is_window=True, window_size=WINDOW_SIZE, stride=STRIDE)

df = loader.load_train_data()
df_test = loader.load_test_data()
df = preprocessor.preprocess_train_data(df, TRAIN_LABEL)
# df_test_normal = preprocessor.preprocess_test_data(df_test, TRAIN_LABEL)
# df_test_neptune = preprocessor.preprocess_test_data(df_test, TEST_LABEL)
(windows, y_label) = preprocessor.preprocess_test_data_multilabel(
    df_test, TRAIN_LABEL, TEST_LABEL)

y_label = np.array(y_label)

model = AutoEncoderModelLSTM(df.shape[2], WINDOW_SIZE, UNITS)
model_executor = AutoencoderModelExecutor(model, epochs=5)
classificator = AutoencoderWindowsResultsClassificator()
analyzer = AutoencoderResultsAnlyzer()

model_executor.fit(df)

x_predicted = model_executor.predict(windows)

classificator.feed(windows, x_predicted, y_label)

error = classificator.calculate_reconstruction_error_windows()

max_f1_score, best_threshold = classificator.calculate_best_threshold()
analyzer.feed(error, y_label, best_threshold,
              max_f1_score, f'{TEST_LABEL}/LSTM_{WINDOW_SIZE}', f'{TEST_LABEL.upper()}, window size = {WINDOW_SIZE}')

analyzer.plot_results()
analyzer.plot_confusion_matrix()
