from Classification.AutoencoderWindowsResultsClassificator import AutoencoderWindowsResultsClassificator
from DataPreparing.KIBANALoader import KIBANALoader
from DataPreparing.KIBANAPreprocessor import KIBANAPreprocessor
from Execution.AutoencoderModelExecutor import AutoencoderModelExecutor
from Model.AutoEncoderModelLSTM import AutoEncoderModelLSTM
import numpy as np
import matplotlib.pyplot as plt
from Analisis.AutoencoderResultsAnlyzer import AutoencoderResultsAnlyzer
from vectorizer_function import get_tokens_for_tfidf

UNITS = [80, 50, 20]
WINDOW_SIZE = 20
STRIDE = 5


TAKE_TOP_ERRORS = 20

loader = KIBANALoader()
preprocessor = KIBANAPreprocessor(
    windows=True, windows_size=WINDOW_SIZE, windows_stride=STRIDE)

df = loader.load_train_data()
df_test = loader.load_test_data()
test_lines = loader.load_test_data_lines()

df = preprocessor.preprocess_train_data(df)
df_test = preprocessor.preprocess_test_data(df_test)


y_label = np.zeros(len(df_test))

model = AutoEncoderModelLSTM(df.shape[2], WINDOW_SIZE, UNITS)
model_executor = AutoencoderModelExecutor(model, epochs=5)
classificator = AutoencoderWindowsResultsClassificator()
analyzer = AutoencoderResultsAnlyzer()

model_executor.fit(df)

x_predicted = model_executor.predict(df_test)

classificator.feed(df_test, x_predicted, y_label)

error = classificator.calculate_reconstruction_error_windows()


def sortSecond(val):
    return val[1]


ordered_errors = list(
    map(lambda e: (e[0], e[1][0]), enumerate(error)))
ordered_errors.sort(key=sortSecond, reverse=True)
highest_errors_indexes = list(map(
    lambda x: x[0], ordered_errors[:TAKE_TOP_ERRORS]))

errors_text = []

for index in highest_errors_indexes:
    print(f'\n\n\nWINDOWS with index nr: {index}')
    errors_text.append(
        f'\n\n\nWINDOWS with index nr: {index}, ERROR SCORE: {error[index]}')
    for i in range(index * STRIDE, index * STRIDE + WINDOW_SIZE):
        print(test_lines[i])
        errors_text.append(test_lines[i])

with open('errors_results.txt', 'a') as the_file:
    for line in errors_text:
        the_file.write(f"{line}")


# error_lines =
# errors = np.concatenate([error_anomaly, error_normal])
# y_label = np.concatenate([[1 for _ in range(len(error_anomaly))],
#                           [0 for _ in range(len(error_normal))]])

# (max_f1_score, best_threshold) = classificator.calculate_best_threshold()

# analyzer.feed(errors, y_label, best_threshold,
#               max_f1_score, f'AE_1/Dense', 'AE_1'.upper())

# analyzer.plot_results()
# analyzer.plot_confusion_matrix()


# print(error_normal)
# print(high_error_indexes)
# print(np.array(df_test.values)[list(map(lambda h: h[0], high_error_indexes))])
