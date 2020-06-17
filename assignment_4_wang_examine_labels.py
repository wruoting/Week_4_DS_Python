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

# This file has been updated with the weekly and daily volatility calculations

def transform_trading_days_to_trading_weeks(df):
    '''
    df: dataframe of relevant data
    returns: dataframe with processed data, only keeping weeks, their open and close for said week
    '''
    trading_list = deque()
    # Iterate through each trading week
    for trading_week, df_trading_week in df.groupby(['Year','Week_Number']):
        week_number =  df_trading_week.iloc[0][['Week_Number']].values[0]
        closing_day_of_week = df_trading_week.iloc[-1][['Close']].values[0]
        trading_list.append([trading_week[0], trading_week[1], closing_day_of_week])
    trading_list_df = pd.DataFrame(np.array(trading_list), columns=['Year', 'Trading Week', 'Week Close'])
    return trading_list_df

def weekly_return_volatility(ticker_file, output_file):
    '''
    ticker_file: string of ticker file name
    output_file: string of output file name
    return: None
    '''
    df = pd.read_csv(ticker_file, encoding='ISO-8859-1')
    df['Return'] = df['Close'].pct_change()
    df['Return'].fillna(0, inplace = True)
    df['Return'] = 100.0 * df['Return']
    df['Return'] = df['Return'].round(3)        
    df_2 = df[['Year', 'Week_Number', 'Return']]
    df_2.index = range(len(df))
    # Mean and Std deviation are in percentages
    df_grouped = df_2.groupby(['Year', 'Week_Number'])['Return'].agg([np.mean, np.std])
    df_grouped.reset_index(['Year', 'Week_Number'], inplace=True)
    df_grouped.rename(columns={'mean': 'mean_return', 'std':'volatility'}, inplace=True)
    df_grouped.fillna(0, inplace=True)

    # After we find the df for weekly mean, we need to find the End of week to End of week returns
    # Some days do not end on a friday
    # Week 1 will not have any returns since we're not going to calculate Friday before that
    df_close_week = transform_trading_days_to_trading_weeks(df)
    df_close_week['Week Close'] = df_close_week['Week Close'].diff()
    df_close_week['Week Close'].fillna(0, inplace = True)
    df_grouped.insert(len(df_grouped.columns), "Abs_Diff", np.abs(df_close_week['Week Close']), allow_duplicates=True)
    df_grouped.insert(len(df_grouped.columns), "Classification", df['Classification'], allow_duplicates=True)

    df_grouped.to_csv(output_file, index=False)

def graph_plot(year, df):
    '''
    year: string of year to plot
    df: dataframe of interest to plot
    return: None
    '''
    # Change our scale to be from 0 - 10 to get bigger (and standardized) points
    scaled_diff = np.divide(df['Abs_Diff'], np.mean(df['Abs_Diff']))
    ax = df.plot.scatter(x='mean_return', y='volatility', s=scaled_diff * 10, c=df['Classification'],  title='WMT Mean to Std Dev {}'.format(year))
    ax.set_xlabel('Mean Weekly Return (%)')
    ax.set_ylabel('Std Dev Weekly Return (%)')
    # Label each point on the graph
    for i, point in df.iterrows():
        ax.text(point['mean_return'], point['volatility'], i, fontsize=6)
    plt.savefig(fname='Examine_Labels_Plot_{}'.format(year))
    plt.show()
    plt.close()

def main():
    ticker='WMT'
    # Requires the self labeled file
    ticker_file = './{}_Labeled_Weeks_Self.csv'.format(ticker)
    output_file = '{}_weekly_return_volatility.csv'.format(ticker)  
    file_name = 'WMT_weekly_return_volatility.csv'
    # Create the output returns of mu and vol of the file
    weekly_return_volatility(ticker_file, output_file)

    # Read from that file for answering our questions
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]
    df_2019.reset_index(inplace=True)

    # Graph each year
    print('You can zoom in and out in matplotlib to see the labels.')
    graph_plot('2018_2019', df)
    graph_plot('2018', df_2018)
    graph_plot('2019', df_2019)


    print('Images to look at are called: Examine_Labels_Plot_2018.png and Examine_Labels_Plot_2019.png')
    print('Both years are in Examine_Labels_Plot_2018_2019.png')
    print('\nQuestion 1:')
    print('The obvious pattern here is that the majority of the points are clustered within one or two standard deviations of the returns ')
    print('and one to two percent within the mean.')
    print('The original labeling was done based on open to close prices, not close to close prices, so therefore there are some points on here ')
    print('which are green when weekly returns are less than 0 and red when they are greater than 0. There are also positive return weeks where we ')
    print('don\'t trade because we considered the volatility to be too high.')
    print('When there are smaller average weekly returns, the dots are smaller, so the size of the dots (weekly return) correlates with average daily return for that week. The ')
    print('greater the average daily return, the greater the weekly return, and the inverse is also true. Most red days have negative average daily returns, but ')
    print('there are some "bad" days with positive average daily returns.')
    print('2018 seems to have some weeks with big volatility, while 2019 did not have some of those high volatility weeks. Higher volatility does not necessarily mean ')
    print('negative returns, but I did choose to not trade on those weeks due to volatility, resulting in more red weeks with high volatility.')

    print('\nQuestion 2:')
    print('Generally, it seems that some green points cluster with green points, and some red points are clustered together as well.')
    print('Mostly, there are red clusters but not too many green ones. We also trade less in general, so there are more red points.')
    print('Still, there are green points that are next to red points and vice versa without significant and obvious clusters.')

    print('\nQuestion 3:')
    print('There are some patterns that are similar, like the grouping of points near 1% of weekly returns, and within 2 std. deviations ')
    print('2018 seems to have some weeks with greater weekly losses than 2019. In addition, the smaller dot size correlates with less variance')
    print('and average daily returns for both years.')

    print('\nQuestion 4:')
    print('I expect nearest neighbors to do well for neither model, since there does not seem to be a set of clusters for ')
    print('either year. There should be some decent overlap based off of the orientation of the clustering within 2 std deviations and a 0% average daily return. ')
    print('Hopefully that consistently in pattern between the two years provides trades that can show good performance.')

if __name__ == "__main__":
    main()