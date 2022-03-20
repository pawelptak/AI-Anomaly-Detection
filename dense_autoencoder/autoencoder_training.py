import os.path
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
import os
import plotly.express

pd.options.plotting.backend = "plotly"


def autoencoder_model(X):
    encoding_dim = 32
    num_sequences = X.shape[1]
    inputs = Input(shape=(num_sequences,))
    encoded = Dense(encoding_dim, activation='relu')(inputs)
    decoded = Dense(num_sequences, activation='sigmoid')(encoded)
    model = Model(inputs=inputs, outputs=decoded)
    return model


def fit_model(model, X_train, y_train, epochs, batch_size, val_split, plot=False):
    callback = EarlyStopping(monitor='loss', patience=4)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=val_split, shuffle=False, callbacks=[callback]).history
    if plot:
        # plot the training losses
        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
        ax.plot(history['loss'], 'b', label='Train', linewidth=2)
        ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
        ax.set_title('Model loss', fontsize=16)
        ax.set_ylabel('Loss (mae)')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')
        plt.show()

    return model


def generate_class_plot(df, col_name, plot_title):
    column = df[col_name]
    classes_list = column.unique()

    y = []
    for c in classes_list:
        y.append(column.value_counts()[c])

    plot_df = pd.DataFrame()
    plot_df['label'] = classes_list
    plot_df['count'] = y
    for i in range(len(classes_list)):  # show values on bars
        plt.text(classes_list[i], y[i] + 1, y[i])

    fig = px.bar(plot_df, x='label', y='count', text='count', title=plot_title)
    fig.show()


# save dataframe to file and reshape data for neural network
def process_train_test_data(df, dataset_name):
    df.to_csv(os.path.join('data/processed', f'{dataset_name}.csv'), index=False)
    #generate_class_plot(df, 'Label', f'{dataset_name} dataset')
    print(f"{dataset_name} data size: {len(df)}")
    df.pop('Label')
    data = np.array(df)
    return data


def get_training_loss(model, X_train):
    predicted = model.predict(X_train)
    train_mae_loss = np.mean(np.abs(predicted - X_train), axis=tuple(range(1, X_train.ndim)))
    train_score_df = pd.DataFrame()
    train_score_df['loss'] = train_mae_loss

    fig = plotly.express.histogram(train_score_df, x="loss", marginal="rug", title='Training loss distribution')
    fig.show()

    return train_mae_loss


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(description='Dense Autoencoder training')

    my_parser.add_argument('Path',
                           metavar='path',
                           type=str,
                           help='Path to the processed csv data file')

    my_parser.add_argument('Percentage',
                           metavar='train %',
                           type=str,
                           help='Train data percentage, e.g 90')
    args = my_parser.parse_args()

    processed_data_path = args.Path
    if not os.path.exists(processed_data_path):
        print('The specified path does not exist.')
        sys.exit()

    all_data = pd.read_csv('data/processed/processed.csv')
    train_percent = float(args.Percentage)/100  # train data percentage

    normal_logs = all_data[all_data['Label'] == 0]
    anomaly_logs = all_data[all_data['Label'] != 0]

    train_size = int(train_percent * len(normal_logs))

    train_data = normal_logs[:train_size]
    test_data = pd.concat([normal_logs[train_size:], anomaly_logs])

    X_train = process_train_test_data(train_data, 'train')
    X_test = process_train_test_data(test_data, 'test')

    model = autoencoder_model(X_train)
    model.compile(optimizer='adam', loss='mae')
    # model.summary()

    # fit the model to the data
    model = fit_model(model, X_train, X_train, epochs=300, batch_size=32, val_split=0.05)

    # get_training_loss(model, X_train)
    model.save('dense_model.h5')
    print('Training finished. Model saved as "dense_model.h5"')
