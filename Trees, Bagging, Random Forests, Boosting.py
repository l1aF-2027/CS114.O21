from sklearn import tree, preprocessing, metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

!git clone https://github.com/l1aF-2027/filecsv

df = pd.read_csv('/content/filecsv/loan_data.csv')
df.shape

df.dropna(inplace=True) 
df


le = LabelEncoder()
is_Category = df.dtypes == object 
category_column_list = df.columns[is_Category].tolist() 
df[category_column_list] = df[category_column_list].apply(lambda col: le.fit_transform(col)) 
df

X = df.drop(columns = 'Loan_Status')
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

tree_model = tree.DecisionTreeClassifier(min_samples_leaf = 4, min_samples_split = 4, random_state=0)
tree_model.fit(X_train, y_train)

y_predict = tree_model.predict(X_test)

from graphviz import Source
from sklearn.tree import export_graphviz
import graphviz
export_graphviz(tree_model, out_file='loan.dot', feature_names=df.columns[:-1], class_names='Loan_Status', impurity=False, filled=True, rounded=True )
Source.from_file("loan.dot")
