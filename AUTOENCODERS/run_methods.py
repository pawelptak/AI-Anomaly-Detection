from Execution.AutoencoderModelExecutor import AutoencoderModelExecutor
from Model.AutoEncoderModelDense import AutoEncoderModelDense_1
from Model.AutoEncoderModelLSTM import AutoEncoderModelLSTM
from Analisis.AutoencoderResultsAnlyzer import AutoencoderResultsAnlyzer
import numpy as np
from Classification.AutoencoderWindowsResultsClassificator import AutoencoderWindowsResultsClassificator
from DataPreparing.KDDLoader import KDDLoader
from DataPreparing.KDDPreprocessor import KDDPreprocessor
from DataPreparing.CICIDSLoaderWindows import CICIDSLoaderWindows
from DataPreparing.CICIDSPreprocessorWindows import CICIDSPreprocessorWindows
from DataPreparing.EVENTSEQUENCELoader import EVENTSEQUENCELoader
from DataPreparing.EVENTSEQUENCEPreprocessor import EVENTSEQUENCEPreprocessor


def get_loader_preprocessor(dataset: str):
    if(dataset == 'KDD'):
        return (KDDLoader(), KDDPreprocessor(is_window=True))
    if(dataset == 'CIC'):
        return (CICIDSLoaderWindows(), CICIDSPreprocessorWindows())
    if(dataset == 'ES'):
        return (EVENTSEQUENCELoader(), EVENTSEQUENCEPreprocessor())

    raise Exception(f'Not supported dataset: {dataset}')


def get_model(model: str, df):
    if(model == 'DENSE'):
        return AutoEncoderModelDense_1(df.shape[2])
    if(model == 'LSTM'):
        return AutoEncoderModelLSTM(df.shape[2], df.shape[1])

    raise Exception(f'Not supported model: {model}')


def run_train(model_name: str, dataset: str):
    loader, preprocessor = get_loader_preprocessor(dataset)
    df = loader.load_train_data()
    df_test = loader.load_test_data()
    df = preprocessor.preprocess_train_data(df)
    (windows, y_label) = preprocessor.preprocess_test_data(df_test)
    y_label = np.array(y_label)

    model = get_model(model_name, df)
    model_executor = AutoencoderModelExecutor(model, epochs=5)
    classificator = AutoencoderWindowsResultsClassificator()
    analyzer = AutoencoderResultsAnlyzer()

    model_executor.fit(df)

    x_predicted = model_executor.predict(windows)

    classificator.feed(windows, x_predicted, y_label)

    error = classificator.calculate_reconstruction_error_windows()

    max_f1_score, best_threshold = classificator.calculate_best_threshold()
    analyzer.feed(error, y_label, best_threshold,
                  max_f1_score, f'{dataset}/{model_name}', f'Model: {model_name}')

    analyzer.plot_results()
    analyzer.plot_confusion_matrix()


def run_predict(model: str, dataset: str, data_predict_path: str):
    loader, preprocessor = get_loader_preprocessor(dataset)
    df = loader.load_predict_data()

    raise Exception("Not Implemented Exception")


def run(mode: str, model: str, dataset: str, data_predict_path=''):
    if(mode == 'TRAIN'):
        return run_train(model, dataset)
    if(model == 'PREDICT'):
        return run_predict(model, dataset, data_predict_path)

    raise Exception(f'Not supported mode: {mode}')
