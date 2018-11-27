import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_high_correlating_columns(df, threshold=0.9, drop_diagonal=True, drop_inverse=True):
    df_corr = df.corr()
    high_corr_cols = np.where(np.abs(df_corr) > threshold)
    col_ixs = zip(high_corr_cols[0], high_corr_cols[1])
    if drop_diagonal:
        col_ixs = [(i, j) for i,j in col_ixs if i != j]
    if drop_inverse:
        col_ixs = set([tuple(sorted(t)) for t in col_ixs])
    return [(df_corr.columns[i], df_corr.columns[j], np.round(df_corr.iloc[i, j], decimals=3)) for i,j in col_ixs]


def get_sparsely_filled_columns(df, threshold=0.9):
    na_relation = df.isna().sum(axis=0) / df.shape[0]
    return na_relation[na_relation > threshold]


def get_numerical_distrbution(column, filter_rate=0):
    column = column.dropna()
    mean = column.mean()
    fig, axs = plt.subplots(1, 2, figsize=(20,5))
    
    # Plot density
    sns.distplot(column, label=column.name, ax=axs[0])
    axs[0].axvline(x=mean, color='r', label='Mean: ' + str(mean))
    axs[0].legend()
    
    # Plot Boxplot
    sns.boxplot(column, ax=axs[1], linewidth=2.5)
    
    plt.show()


def get_categorical_distrbution(column, filter_rate=0.01):
    fig, axs = plt.subplots(1, 2, figsize=(20,5))
    
    # Plot non filtered values
    counts = column.value_counts()
    counts = counts/sum(counts)
    counts.plot(kind='bar', ax=axs[0], title="Non-filtered bar plot for '" + column.name + "'")
    
    # Plot filtered values
    counts = counts.drop(counts[counts < filter_rate].index)
    counts['__Filtered_Out__'] = 1 - sum(counts)
    counts.plot(kind='bar', ax=axs[1], title="Filtered bar plot for '" + column.name + "' with filter_rate=" + str(filter_rate))
    
    plt.show()