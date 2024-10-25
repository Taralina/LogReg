import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import streamlit as st

st.header('Логистическая регрессия', divider=True)

uploaded_file_train = st.sidebar.file_uploader("Выберите CSV файл для обучения модели", type="csv")

st.write("""
    #### 1. Данные для обучения
         """)

if uploaded_file_train is not None:
    train = pd.read_csv(uploaded_file_train)
    st.write(train)
else:
    train = pd.read_csv('credit_train.csv')    


uploaded_file_test = st.sidebar.file_uploader("Выберите CSV файл для предсказания", type="csv")

st.write("""
    #### 2. Данные для тестирования модели
         """)

if uploaded_file_test is not None:
    test = pd.read_csv(uploaded_file_test)
    st.write(test)
else:
    test = pd.read_csv('credit_test.csv') 


ss = StandardScaler()
X_train = ss.fit_transform(train[['CCAvg', 'Income']])
y_train = train['Personal.Loan']
X_test = ss.transform(test[['CCAvg', 'Income']])
y_test = test['Personal.Loan']


class LogReg:
     def __init__(self, learning_rate, n_inputs):
         self.learning_rate = learning_rate
         self.n_inputs = n_inputs
         self.coef_ = np.random.uniform(low=-1.0, high=1.0, size=n_inputs)
         self.intercept_ = np.random.uniform(low=-1.0, high=1.0)
         
     def sigmoid(self, z):        
         return 1 / (1 + np.exp(-z))
        
     def fit(self, X, y):
         epochs=1000
         for _ in range(epochs):
             linear_output = np.dot(X, self.coef_) + self.intercept_
             y_pred = self.sigmoid(linear_output)

             dw = (1 / len(y)) * np.dot(X.T, (y_pred - y))
             db = (1 / len(y)) * np.sum(y_pred - y)
         
             self.coef_ -= self.learning_rate * dw
             self.intercept_ -= self.learning_rate * db

         return self.coef_, self.intercept_ 

     def predict(self, X):
         linear_output = np.dot(X, self.coef_) + self.intercept_
         return self.sigmoid(linear_output)
     
my_model = LogReg(learning_rate=0.1, n_inputs=2)     

st.write("""
    #### 3. Результат регресии - оптимальные весовые коэффициенты
         """)
my_model.fit(X_train, y_train)
features = ['CCAvg', 'Income']
weights = dict(zip(features, my_model.coef_))
st.write(weights)

w1, w2 = my_model.coef_
b = my_model.intercept_

st.write("""
    #### 4. Предсказания (predict)
         """)
y_pred = my_model.predict(X_train)
y_pred

st.write("""
    #### 5. Сравним результаты со встроенным методом
         """)

lr = LogisticRegression()
lr.fit(X_train, y_train)
weights_lr = {features[0]: lr.coef_[0][0], features[1]:lr.coef_[0][1]}
st.write(weights_lr)

st.write("""
    #### 6. Построим скатерплот по нашим данным
         """)
x_values = np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100)
y_values = - (w1 / w2) * x_values - (b / w2)


fig, ax = plt.subplots()
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolor='k', s=100)
ax.set_xlabel('CCAvg')
ax.set_ylabel('Income')

ax.set_ylim(-2, 2)  
ax.set_xlim(-1.5, 3.5)  
ax.plot(x_values, y_values, color='black', linewidth=2)

cbar = plt.colorbar(scatter, ticks=[0, 1])
cbar.set_label('Personal Loan')
cbar.ax.set_yticklabels(['No', 'Yes'])

st.pyplot(fig)
