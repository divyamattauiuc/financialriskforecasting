from operator import truediv
from turtle import color, ycor
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
from pandas_datareader import data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.ensemble import AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import (Sequential, Model)
from tensorflow.keras.layers import (Dense, Dropout)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             classification_report)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import missingno as msno 

# Standardizing the style for the visualizations 
sns.set_theme()
sns.set(font_scale=1.2)
sns.set_palette("pastel")
plt.style.use('seaborn-whitegrid')

loans = pd.read_csv("/Users/divyamatta/Downloads/accepted-15000.csv")

trim = ["annual_inc", "loan_amnt", "total_bc_limit","avg_cur_bal","pct_tl_nvr_dlq","mo_sin_old_rev_tl_op","tax_liens","fico_range_high","dti", "loan_status","addr_state"]
y = loans["loan_status"]
loans["loan_status_og"] = y
y = y.map({"Fully Paid":0,"Current":2,"Charged Off":1, "In Grace Period":3, "Late (31-120 days)":4, "Late (16-30 days)":5})

print(y)
loans["loan_status"] = y

loans_trim = loans[trim]


# ## Types of Loan Count
# fig, ax =plt.subplots(figsize=(10,10))
# sns.despine()
# sns.countplot(data=loans,x="loan_status")
# ax.set(xlabel='Status', ylabel='')
# ax.set_title('Loan status count', size=20)

## Heat Map

corr=loans_trim.loc[:,loans_trim.columns!="loan_status"].corr()
mask= np.triu(np.ones_like(corr,dtype=np.bool))
fig,ax= plt.subplots(figsize=(15,10))
sns.heatmap(corr,mask=mask, annot=True, fmt=".2f",cbar_kws={"shrink": .8}, vmin=0, vmax=1)

target_loan= ["Fully Paid"]
map_fp=loans[loans["loan_status"].isin(target_loan)]
m=map_fp[["addr_state","loan_status"]]
m=m.groupby(["addr_state"])["loan_status"].agg('count').reset_index()

# Bar Graph State
# fig, ax =plt.subplots(figsize=(20,10))
# sns.despine()
# order = loans_trim["addr_state"].value_counts().index
# sns.countplot(data=loans_trim,x="addr_state",order=order)
# ax.tick_params(axis='x', labelrotation=90)
# ax.set(xlabel='State', ylabel='')
# ax.set_title('Loan count by state', size=20)

# Map Graph State FULLY PAID


# target_loan= ["Fully Paid"]
# map_fp=loans[loans["loan_status_og"].isin(target_loan)]
# m=map_fp[["addr_state","loan_status_og"]]
# m=m.groupby(["addr_state"])["loan_status_og"].agg('count').reset_index()

# fig = go.Figure(data=go.Choropleth(
#     locations=m["addr_state"], # Spatial coordinates
#     z = m["loan_status_og"].astype(float), # Data to be color-coded
#     locationmode = 'USA-states', # set of locations match entries in `locations`
#     colorscale = 'bluyl',
#     colorbar_title = "Fully Paid Loans",

# )
# )
# print(m["loan_status_og"])
# print(m["loan_status_og"].astype(float))
# fig.update_layout(
#     title_text = 'Fully Paid Loans 2008-2018 timeframe',
#     geo_scope='usa', # limit the map scope to USA
# )
# Map Graph State Charged Off

# target_loan= ["Charged Off"]
# map_co=loans[loans["loan_status_og"].isin(target_loan)]
# m=map_co[["addr_state","loan_status_og"]]
# m=m.groupby(["addr_state"])["loan_status_og"].agg('count').reset_index()

# fig = go.Figure(data=go.Choropleth(
#     locations=m["addr_state"], # Spatial coordinates
#     z = m["loan_status_og"].astype(float), # Data to be color-coded
#     locationmode = 'USA-states', # set of locations match entries in `locations`
#     colorscale = 'Reds',
#     colorbar_title = "Charged Off Loans",
# ))

# fig.update_layout(
#     title_text = 'Charged Off Loans 2008-2018 timeframe',
#     geo_scope='usa', # limit the map scope to USA
# )

## Box Plot
# Installment amount count by loan status


# fig, ax =plt.subplots(1,2,figsize=(20,8))

# sns.despine() 

# ax[0].tick_params(axis='x', labelrotation=0)
# ax[0].set(xlabel='Installments amount in USD', ylabel='')
# ax[0].set_title('Installment amount by loan type - Distribution', size=20)
# ax[1].tick_params(axis='x', labelrotation=0)
# ax[1].set_title('Installment amount by loan type - Boxplot', size=20)


# sns.histplot(data=loans,x="installment",hue="loan_status",bins=30,
#             kde=True,ax=ax[0])
# sns.boxplot(data=loans,x="loan_status",y="installment",ax=ax[1]).set(xlabel='Loan Status', 
#                                                                        ylabel='Amount in USD')


## Box Plot 2 
# Loan amount count by loan status
# fig, ax =plt.subplots(1,2,figsize=(20,8))

# sns.despine() 

# ax[0].tick_params(axis='x', labelrotation=0)
# ax[0].set(xlabel='Annual Income in USD', ylabel='')
# ax[0].set_title('Annual Income by loan type - Distribution', size=20)
# ax[1].tick_params(axis='x', labelrotation=0)
# ax[1].set_title('Annual Income by loan type - Boxplot', size=20)

# sns.histplot(data=loans,x="annual_inc",hue="loan_status",bins=15,
#             kde=True,ax=ax[0])
# sns.boxplot(data=loans,x="annual_inc",y="loan_amnt",ax=ax[1]).set(xlabel='Loan Status', 
#                                                                        ylabel='Amount in USD')

plt.show()
fig.show()