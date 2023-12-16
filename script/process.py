import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import Image
from plotly.subplots import make_subplots
def get_df_percent_missing(df: pd.DataFrame) -> str:
    totalCells = np.product(df.shape)
    missingCount = df.isnull().sum()
    totalMissing = missingCount.sum()
    return f"The telecom contains {round(((totalMissing/totalCells) * 100), 2)}% missing values."

def get_df_discribe(df:pd.DataFrame)->pd.DataFrame:
    return df.describe()

def get_df_null_count(df:pd.DataFrame)->pd.DataFrame:
    return df.isna().sum()

def get_df_information(df:pd.DataFrame)->pd.DataFrame:
    return df.info()

def get_missing_colum_percentage(df: pd.DataFrame) -> pd.DataFrame:
    num_missing = df.isnull().sum()
    num_rows = df.shape[0]

    data = {
        'num_missing': num_missing, 
        'percent_missing (%)': [round(x, 2) for x in num_missing / num_rows * 100]
    }

    stats = pd.DataFrame(data)

    # Filter columns with missing values
    return stats[stats['num_missing'] != 0]


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str,title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data = df, x=x_col, y=y_col)
    plt.title(f'{title}')
    plt.xticks(fontsize=14)
    plt.yticks( fontsize=14)
    plt.show()

def plot_heatmap(df:pd.DataFrame, title:str, cbar=False)->None:
    plt.figure(figsize=(12, 7))
    sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
    plt.title(title, size=18, fontweight='bold')
    plt.show()

def plot_hist(df:pd.DataFrame, column:str, color:str)->None:
    # plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, figsize=(12, 7))
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of {column}', size=20, fontweight='bold')
    plt.show()

def mult_hist(sr, rows, cols, title_text, subplot_titles, interactive=False):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    for i in range(rows):
        for j in range(cols):
            x = ["-> " + str(i) for i in sr[i+j].index]
            fig.add_trace(go.Bar(x=x, y=sr[i+j].values), row=i+1, col=j+1)
    fig.update_layout(showlegend=False, title_text=title_text)
    if(interactive):
        fig.show()
    else:
        return Image(pio.to_image(fig, format='png', width=1200))