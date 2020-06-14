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

def label_points(df_year, year, name):
    '''
    df_year: df of points that we need labeled
    year: the year of the labeled points
    name: name of the points
    return: None
    '''
    # Change our scale to be from 0 - 10
    scaled_diff = np.divide(df_year['Abs_Diff'], np.mean(df_year['Abs_Diff']))
    ax = df_year.plot.scatter(x='mean_return', y='volatility', s=scaled_diff * 10, c=df_year['Classification'],  title='WMT Mean to Std Dev {}'.format(year))
    ax.set_xlabel('Mean Weekly Return (%)')
    ax.set_ylabel('Std Dev Weekly Return (%)')
    for i, point in df_year.iterrows():
        ax.text(point['mean_return'], point['volatility'], i, fontsize=6)
    plt.savefig(fname=name)
    plt.show()
    plt.close()

def label_points_add_line(df_year, year, name):
    '''
    df_year: df of points that we need labeled
    year: the year of the labeled points
    name: name of the points
    return: None
    '''
    # Add a line
    # Change our scale to be from 0 - 10
    x = np.linspace(-3, 3, 100)
    scaled_diff = np.divide(df_year['Abs_Diff'], np.mean(df_year['Abs_Diff']))
    ax1 = df_year.plot.scatter(x='mean_return', y='volatility', s=scaled_diff * 10, c=df_year['Classification'],  title='WMT Mean to Std Dev With Line {}'.format(year))
    ax1.set_xlabel('Mean Weekly Return (%)')
    ax1.set_ylabel('Std Dev Weekly Return (%)')
    ax2 = plt.plot(x, 5*x+1.7)
    plt.savefig(fname=name)
    plt.show()
    plt.close()


def transform_trading_days_to_trading_weeks(df):
    '''
    df: dataframe of relevant data
    returns: dataframe with processed data, only keeping weeks, their open and close for said week
    '''
    trading_list = deque()
    # Iterate through each trading week
    for trading_week, df_trading_week in df.groupby(['Year','Week_Number']):
        classification =  df_trading_week.iloc[0][['Classification']].values[0]
        opening_day_of_week = df_trading_week.iloc[0][['Open']].values[0]
        closing_day_of_week = df_trading_week.iloc[-1][['Close']].values[0]
        trading_list.append([trading_week[0], trading_week[1], opening_day_of_week, closing_day_of_week, classification])
    trading_list_df = pd.DataFrame(np.array(trading_list), columns=['Year', 'Trading Week', 'Week Open', 'Week Close', 'Classification'])
    return trading_list_df

def make_trade(cash, open, close):
    '''
    cash: float of cash on hand
    open: float of open price
    close: float of close price
    returns: The cash made from a long position from open to close
    '''
    shares = np.divide(cash, open)
    return np.multiply(shares, close)

def trading_strategy(trading_df, is_predicted, weekly_balance=100):
    '''
    trading_df: dataframe of relevant weekly data
    returns: A df of trades made based on Predicted Labels
    '''
    if is_predicted:
        labels = 'Predicted Labels'
    else:
        labels = 'Classification'
    # The weekly balance we will be using
    weekly_balance_acc = weekly_balance
    trading_history = deque()
    index = 0
    
    while(index < len(trading_df.index) - 1):
        trading_week_index = index
        if weekly_balance_acc != 0:
            # Find the next consecutive green set of weeks and trade on them
            while(trading_week_index < len(trading_df.index) - 1 and trading_df.iloc[trading_week_index][[labels]].values[0] == 'GREEN'):
                trading_week_index += 1
            green_weeks = trading_df.iloc[index:trading_week_index][['Week Open', 'Week Close']]
            # Check if there are green weeks, and if there are not, we add a row for trading history
            if len(green_weeks.index) > 0:
                # Buy shares at open and sell shares at close of week
                green_weeks_open = float(green_weeks.iloc[0][['Week Open']].values[0])
                green_weeks_close = float(green_weeks.iloc[-1][['Week Close']].values[0])
                # We append the money after we make the trade
                weekly_balance_acc = make_trade(weekly_balance_acc, green_weeks_open, green_weeks_close)
            # Regardless of whether we made a trade or not, we append the weekly cash and week over
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                weekly_balance_acc])
        else:
            # If we have no money we will not be able to trade
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                    trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                    weekly_balance_acc])
        index = trading_week_index+1
    trading_hist_df = pd.DataFrame(np.array(trading_history), columns=['Year', 'Trading Week', 'Balance'])
    trading_hist_df['Balance'] = np.round(trading_hist_df[['Balance']].astype(float), 2)

    return trading_hist_df

def main():
    ticker='WMT'
    file_name = '{}_weekly_return_volatility.csv'.format(ticker)
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'

    # Read from that file for answering our questions
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]
    
    print('\nQuestion 1')
    print('Label points in text to allow removal')
    label_points(df_2018, '2018', 'Separate_Points_Plot_2018_Pre')
    print('Remove points to allow line to go through')
    print('This function adds labels and is called Separate_Points_Plot_2018 so that I can remove points')
    print('The following indices are removed from 2018: 51, 11, 13, 39, 40, 43, 22, 21, 41, 47, 27, 44, 6, 52, 19, 25, 34, 42, 37, 24, 0, 23, 35')
    delete_2018 = [51, 11, 13, 39, 40, 43, 22, 21, 41, 47, 27, 44, 6, 52, 19, 25, 34, 42, 37, 24, 0, 23, 35]
    df_2018 = df_2018.drop(df_2018.index[delete_2018].copy())
    print('This is the post labeled points')
    label_points(df_2018, '2018', 'Separate_Points_Plot_2018_Post')
    print('Now we draw the line')
    label_points_add_line(df_2018, '2018', 'Separate_Points_Plot_2018_With_Line')
    
    print('\nQuestion 2')
    print('Assign labels for equation y = 5x+1.7')
    df_y_predicted = 5*df_2019[['mean_return']]+1.7
    df_y_predicted.rename(columns={ 'mean_return' : 'volatility' }, inplace=True)
    predicted_classifications = np.where(df_y_predicted >= df_2019[['volatility']], 'GREEN', 'RED')
    print('Year Labels are called Predicted Labels:')
    print(predicted_classifications.T)

    print('\nQuestion 3')
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_trading_weeks = transform_trading_days_to_trading_weeks(df)
    trading_weeks_2019 = df_trading_weeks[df_trading_weeks['Year'] == '2019']
    trading_weeks_2019.reset_index(inplace=True)
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Predicted Labels", predicted_classifications, allow_duplicates=True)
    print('Trading Strategy for 2019 for $100 starting cash:')
    print('Trading strategy was based on the one created in Assignment 3')
    print('With Predicted Labels:')
    predicted_trading_df = trading_strategy(trading_weeks_2019, True)
    print('${}'.format(predicted_trading_df[['Balance']].iloc[-1].values[0]))
    print('With Self Classifications')
    classification_trading_df = trading_strategy(trading_weeks_2019, False)
    print('${}'.format(classification_trading_df[['Balance']].iloc[-1].values[0]))
if __name__ == "__main__":
    main()