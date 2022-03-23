from Classification.AutoencoderResultsClassificator import AutoencoderResultsClassificator
from DataPreparing.KDDLoader import KDDLoader
from DataPreparing.KDDPreprocessor import KDDPreprocessor
from Execution.AutoencoderModelExecutor import AutoencoderModelExecutor
from Model.AutoEncoderModelDense import AutoEncoderModelDense_1, AutoEncoderModelDense_2
import numpy as np
import matplotlib.pyplot as plt


SERVICE = "http"
TRAIN_LABEL = "normal"
TEST_LABEL = "neptune"  # back
#TRAIN_LABEL = 0
#TEST_LABEL = 1

loader = KDDLoader(SERVICE)
preprocessor = KDDPreprocessor()

df = loader.load_train_data()
df_test = loader.load_test_data()
df = preprocessor.preprocess_train_data(df, TRAIN_LABEL)
df_test_normal = preprocessor.preprocess_test_data(df_test, TRAIN_LABEL)
df_test_neptune = preprocessor.preprocess_test_data(df_test, TEST_LABEL)

tests = range(1, df.shape[1] + 1)
size = len(tests)

dense_sizes = tests
dense_sizes_scores = np.zeros(size)

for index, dense_size in enumerate(dense_sizes):
    max_score = 0
    for i in range(5):
        model = AutoEncoderModelDense_1(df.shape[1], dense_size)
        model_executor = AutoencoderModelExecutor(model, epochs=3)
        classificator = AutoencoderResultsClassificator()

        model_executor.fit(df.values)

        x_normal = df_test_normal.values
        x_anomaly = df_test_neptune.values
        y_normal = model_executor.predict(x_normal)
        y_anomaly = model_executor.predict(x_anomaly)

        classificator.feed(x_normal, y_normal, x_anomaly, y_anomaly)

        (error_normal, error_anomaly) = classificator.calculate_reconstruction_error()

        (max_f1_score, best_threshold) = classificator.calculate_best_threshold()

        if(max_f1_score > max_score):
            max_score = max_f1_score

    dense_sizes_scores[index] = max_score

np.save("Results/AE_2_results/neptune_scores",
        [dense_sizes, dense_sizes_scores])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("NEPTUNE")
plt.plot(dense_sizes, dense_sizes_scores)
plt.ylabel('F - score')
plt.xlabel('S')
plt.ylim(0, 1)


SERVICE = "http"
TRAIN_LABEL = "normal"
TEST_LABEL = "back"  # back
#TRAIN_LABEL = 0
#TEST_LABEL = 1

loader = KDDLoader(SERVICE)
preprocessor = KDDPreprocessor()

df = loader.load_train_data()
df_test = loader.load_test_data()
df = preprocessor.preprocess_train_data(df, TRAIN_LABEL)
df_test_normal = preprocessor.preprocess_test_data(df_test, TRAIN_LABEL)
df_test_neptune = preprocessor.preprocess_test_data(df_test, TEST_LABEL)

tests = range(1, df.shape[1] + 1)
size = len(tests)

dense_sizes = tests
dense_sizes_scores = np.zeros(size)

for index, dense_size in enumerate(dense_sizes):
    max_score = 0
    for i in range(5):
        model = AutoEncoderModelDense_1(df.shape[1], dense_size)
        model_executor = AutoencoderModelExecutor(model, epochs=3)
        classificator = AutoencoderResultsClassificator()

        model_executor.fit(df.values)

        x_normal = df_test_normal.values
        x_anomaly = df_test_neptune.values
        y_normal = model_executor.predict(x_normal)
        y_anomaly = model_executor.predict(x_anomaly)

        classificator.feed(x_normal, y_normal, x_anomaly, y_anomaly)

        (error_normal, error_anomaly) = classificator.calculate_reconstruction_error()

        (max_f1_score, best_threshold) = classificator.calculate_best_threshold()

        if(max_f1_score > max_score):
            max_score = max_f1_score

    dense_sizes_scores[index] = max_score

np.save("Results/AE_2_results/back_scores", [dense_sizes, dense_sizes_scores])

plt.subplot(1, 2, 2)
plt.plot(dense_sizes, dense_sizes_scores)
plt.title("BACK")
plt.ylabel('F - score')
plt.xlabel('S')
plt.ylim(0, 1)
plt.savefig("Results/AE_2_results/scores.png")
