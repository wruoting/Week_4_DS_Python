import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# run this  !pip install pandas_datareader
import os
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from assignment_4_wang_separate_points import make_trade, transform_trading_days_to_trading_weeks, trading_strategy

def main():
    ticker='WMT'
    file_name = '{}_weekly_return_volatility.csv'.format(ticker)
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'

    # Read from that file for answering our questions
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]

    print('\nQuestion 1')
    X = df_2018[['mean_return', 'volatility']].values
    Y = df_2018[['Classification']].values

    error_rate = {}
    for n in range(3, 13, 2):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, random_state=3)
        # KNN Classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=n)
        knn_classifier.fit(X_train, Y_train.ravel())
        prediction = knn_classifier.predict(X_test)
        # As a percentage
        error_rate[n] = np.round(np.multiply(np.mean(prediction != Y_test), 100), 2)
    plt.plot(list(error_rate.keys()), list(error_rate.values()))
    plt.title('Number of Nearest Neighbors vs Error rate')
    plt.xlabel('Number of Nearest Neighbors')
    plt.ylabel('Error Rate (%)')
    plt.savefig(fname='KNN_Classifiers_Q1')
    plt.show()
    plt.close()
    print('Error rate is {}'.format(error_rate))
    print('Lowest error is 9 nearest neighbors, so that is the optimal k value for year 1')

    print('\nQuestion 2')
    X_2019 = df_2019[['mean_return', 'volatility']].values
    Y_2019 = df_2019[['Classification']].values
    knn_classifier = KNeighborsClassifier(n_neighbors=9)
    knn_classifier.fit(X, Y.ravel())
    prediction = knn_classifier.predict(X_2019)
    accuracy_2019 = np.round(np.multiply(np.mean(prediction == Y_2019), 100), 2)
    print('Accuracy: {}%'.format(accuracy_2019))

    print('\nQuestion 3')
    confusion_matrix_array = confusion_matrix(Y_2019, prediction)
    confusion_matrix_df = pd.DataFrame(confusion_matrix_array, columns= ['Predicted: GREEN', 'Predicted: RED'], index=['Actual: GREEN', 'Actual: RED'])
    print(confusion_matrix_df)

    print('\nQuestion 4')
    total_data_points = len(Y_2019)
    true_positive_number = confusion_matrix_df['Predicted: GREEN']['Actual: GREEN']
    true_positive_rate = np.round(np.multiply(np.divide(true_positive_number, total_data_points), 100), 2)
    true_negative_number = confusion_matrix_df['Predicted: RED']['Actual: RED']
    true_negative_rate = np.round(np.multiply(np.divide(true_negative_number, total_data_points), 100), 2)
    print("True positive rate: {}%".format(true_positive_rate))
    print("True negative rate: {}%".format(true_negative_rate))

    print('\nQuestion 5')
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_trading_weeks = transform_trading_days_to_trading_weeks(df)
    trading_weeks_2019 = df_trading_weeks[df_trading_weeks['Year'] == '2019']
    buy_and_hold = np.full(len(trading_weeks_2019.index), 'GREEN')
    trading_weeks_2019.reset_index(inplace=True)
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Predicted Labels", prediction, allow_duplicates=True)
    # Create only green trades to simulate buy and hold
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Buy and Hold", buy_and_hold, allow_duplicates=True)

    predicted_trading_df = trading_strategy(trading_weeks_2019, "Predicted Labels")
    predicted_trading_buy_and_hold = trading_strategy(trading_weeks_2019, "Buy and Hold")
    print('KNN Model')
    print('${}'.format(predicted_trading_df[['Balance']].iloc[-1].values[0]))
    print('Buy and Hold')
    print('${}'.format(predicted_trading_buy_and_hold[['Balance']].iloc[-1].values[0]))
    print('The buy and hold strategy earns more than this KNN model.')

if __name__ == "__main__":
    main()