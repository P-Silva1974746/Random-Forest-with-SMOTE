from random_forest import RandomForestClassifier
from metrics import accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
import openml
from openml import tasks


# # Small test with a local dataset just to make sure the code is running 

# df=pd.read_csv("weather.csv")
# n_col=df.columns.size
# df['Play'].replace({'no': 0,'yes':1}, inplace=True)
# Y= df.iloc[:, n_col-1]
# X= df.iloc[:, 1:n_col-2]


# X_train, X_test, Y_train, Y_test= train_test_split(X,Y)
# print(X_train)
# print(Y_train)
# print('''


# ''')
# model=RandomForestClassifier()
# model.fit(X_train,Y_train)
# predictions=model.predict(X_test)
# print(X_test)
# print(f"Accuracy: {accuracy(Y_test,predictions)}")


benchmark=openml.study.get_suite(suite_id=99)
print('''Getting datasets
''')
datasets_info_df=tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
for id in datasets_info_df["did"]:
    print(id)

    dataset=openml.datasets.get_dataset(id)
    X,Y,_,_=dataset.get_data(target=dataset.default_target_attribute , dataset_format="dataframe")
    
    X_train, X_test, Y_train, Y_test =train_test_split(X,Y)

    model = RandomForestClassifier(max_depth=15)
    model.fit(X_train,Y_train)
    predictions=model.predict(X_test)

    print(f"Accuracy: {accuracy(Y_test,predictions)}")

    print("")
    print("Spot here for checkpoint each dataset")

