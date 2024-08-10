from turtle import color
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
loans = pd.read_csv("credit_risk_dataset.csv")
# loans.Loan_ID.value_counts(dropna=False)
# loans.Gender.value_counts(dropna=False)
# loans.Self_Employed.value_counts(dropna=False)
# loans.Credit_History.value_counts(dropna=False)
# loans.Property_Area.value_counts(dropna=False)
# loans.Loan_Status.value_counts(dropna=False)
# loans.Loan_Amount_Term.value_counts(dropna=False)
# loans.isnull().sum()
# loans['LoanAmount'].fillna(loans['LoanAmount'].mean(),inplace=True)



print(loans)
x = loans[["loan_percent_income","person_income", "cb_person_cred_hist_length"]]
x = x.values
##x = x.reshape(-1,1)


y = loans["cb_person_default_on_file"]


def changer(i):
    if i == "N":
        return 0
    elif i == "Y":
        return 1

y = y.map({"Y":1,"N":0})
print(y)
##y = y.to_numpy()
##y = y.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.1)
##print(x_train.shape)
plt.scatter(x[:,0],y, color = "red")
plt.scatter(x[:,1],y, color = "blue")


# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
# ##print(model.score(X_train, y_train))
# y_pred = model.predict(X_test)
# print(y_pred)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(y_pred, y_test)
##print(rfc.score(y_pred,y_test))
print(y_pred)
print(rfc.predict_proba(X_test))

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# model = LinearRegression()
# model.fit(x_train,y_train)
# y_predict = model.predict(x_train)
# print(y_predict.shape)

# plt.scatter(x_train[:,0],y_predict, color = 'red')
# plt.scatter(x_train[:,1],y_predict, color = 'purple')
# plt.scatter(x_train[:,2],y_predict, color = 'pink')

# value= model.predict([[200,200,200]])
# plt.scatter([200], value, color = 'black')
# plt.scatter([200], value, color = 'black')
# plt.scatter([200], value, color = 'black')
# print(model.coef_)
plt.show()
