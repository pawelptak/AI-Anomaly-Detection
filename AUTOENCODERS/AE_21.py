from Classification.AutoencoderResultsClassificator import AutoencoderResultsClassificator
from DataPreparing.CICIDSLoader import CICIDSLoader
from DataPreparing.CICIDSPreprocessor import CICIDSPreprocessor
from Execution.AutoencoderModelExecutor import AutoencoderModelExecutor
from Model.AutoEncoderModelDense import AutoEncoderModelDense_1, AutoEncoderModelDense_2
from Analisis.AutoencoderResultsAnlyzer import AutoencoderResultsAnlyzer
import numpy as np
import matplotlib.pyplot as plt


ATTACK = "DDoS"
NORMAL = "BENIGN"


loader = CICIDSLoader(ATTACK)
preprocessor = CICIDSPreprocessor()

df = loader.load_train_data()
df_test = loader.load_test_data()

df = preprocessor.preprocess_train_data(df, NORMAL)
df_test_normal = preprocessor.preprocess_test_data(df_test, NORMAL)
df_test_attack = preprocessor.preprocess_test_data(df_test, ATTACK)

model = AutoEncoderModelDense_1(df.shape[1], 15)
model_executor = AutoencoderModelExecutor(model, epochs=5)
classificator = AutoencoderResultsClassificator()
analyzer = AutoencoderResultsAnlyzer()

model_executor.fit(df.values)

x_normal = df_test_normal.values
x_anomaly = df_test_attack.values
y_normal = model_executor.predict(x_normal)
y_anomaly = model_executor.predict(x_anomaly)

classificator.feed(x_normal, y_normal, x_anomaly, y_anomaly)

(error_normal, error_anomaly) = classificator.calculate_reconstruction_error()

(max_f1_score, best_threshold) = classificator.calculate_best_threshold()

errors = np.concatenate([error_anomaly, error_normal])
y_label = np.concatenate([[1 for _ in range(len(error_anomaly))], [
                         0 for _ in range(len(error_normal))]])

# (max_f1_score, best_threshold) = classificator.calculate_best_threshold()

analyzer.feed(errors, y_label, 0, 0, f'{ATTACK}/Dense', "DDoS")

analyzer.plot_results()
# analyzer.plot_confusion_matrix()
