import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from autoencoder_training import get_training_loss
import argparse
import sys
import plotly.express

pd.options.plotting.backend = "plotly"


def load_data(file_name):
    df = pd.read_csv(os.path.join('data/processed', file_name))
    labels = df.pop('Label')
    data = np.array(df)

    return df, data, labels


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Dense Autoencoder training')

    my_parser.add_argument('Model',
                           metavar='model file',
                           type=str,
                           help='Path to the model .h5 file')

    my_parser.add_argument('-t', action='store', type=int, required=False)

    args = my_parser.parse_args()

    if not os.path.exists(args.Model):
        print('The specified model path does not exist.')
        sys.exit()

    model = load_model(args.Model)

    train_df, X_train, train_labels = load_data('train.csv')
    test_df, X_test, test_labels = load_data('test.csv')

    train_mae_loss = get_training_loss(model, X_train)

    threshold = 0
    if args.t:
        threshold = args.t
    else:
        threshold = np.mean(train_mae_loss)  # adjust threshold for better results

    predicted = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(predicted - X_test), axis=tuple(range(1, X_test.ndim)))
    test_score_df = test_df.copy()
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = (test_score_df['loss'] > test_score_df['threshold']) * 1
    test_score_df['actual value'] = test_labels
    test_score_df['actual value'] = test_score_df['actual value'].astype(str)  # for discrete color in plotly

    # with pd.option_context('display.max_rows', None):  # more options can be specified also
    #     print(test_score_df[['anomaly', 'actual value', 'loss', 'threshold']])

    import plotly.graph_objects as go

    fig = plotly.express.scatter(test_score_df, color='actual value', x=test_score_df.index, y='loss', title='Test data loss')
    fig.add_trace(go.Scatter(x=test_score_df.index, y=test_score_df['threshold'], name='threshold'))
    fig.show()

    conf_matrix = confusion_matrix(test_score_df['actual value'].astype(int), test_score_df['anomaly'])
    fig = plotly.express.imshow(conf_matrix, x=['No anomaly', 'Anomaly'], y=['No anomaly', 'Anomaly'], text_auto=True,
                    labels=dict(x="Predicted Values", y="Actual Values"))
    fig.show()
