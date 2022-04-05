from Classification.AutoencoderResultsClassificator import AutoencoderResultsClassificator
from DataPreparing.KIBANALoader import KIBANALoader
from DataPreparing.KIBANAPreprocessor import KIBANAPreprocessor
from Execution.AutoencoderModelExecutor import AutoencoderModelExecutor
from Model.AutoEncoderModelDense import AutoEncoderModelDense_1, AutoEncoderModelDense_2
import numpy as np
import matplotlib.pyplot as plt
from Analisis.AutoencoderResultsAnlyzer import AutoencoderResultsAnlyzer


loader = KIBANALoader()
preprocessor = KIBANAPreprocessor()

df = loader.load_train_data()
df_test = loader.load_test_data()
df = preprocessor.preprocess_train_data(df)
df_test = preprocessor.preprocess_test_data(df_test)

model = AutoEncoderModelDense_1(df.shape[1], 20)
model_executor = AutoencoderModelExecutor(model, epochs=5)
classificator = AutoencoderResultsClassificator()
analyzer = AutoencoderResultsAnlyzer()

model_executor.fit(df.values)

x_normal = df_test.values
x_anomaly = df_test.values
y_normal = model_executor.predict(x_normal)
y_anomaly = model_executor.predict(x_anomaly)

classificator.feed(x_normal, y_normal, x_anomaly, y_anomaly)

(error_normal, error_anomaly) = classificator.calculate_reconstruction_error()

errors = np.concatenate([error_anomaly, error_normal])
y_label = np.concatenate([[1 for _ in range(len(error_anomaly))],
                          [0 for _ in range(len(error_normal))]])

(max_f1_score, best_threshold) = classificator.calculate_best_threshold()

analyzer.feed(errors, y_label, best_threshold,
              max_f1_score, f'AE_1/Dense', 'AE_1'.upper())

analyzer.plot_results()
analyzer.plot_confusion_matrix()
