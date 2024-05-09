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

def encod_dict(X):
    dic={}
    values=X.unique()
    num=0
    for value in values:
        dic.update({value: num})
        num+=1
    
    return dic



benchmark=openml.study.get_suite(suite_id=99)
print('''Getting datasets ...
''')
datasets_info_df=tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
f = open("datasets_used.txt","w")
for index, dataset_info in datasets_info_df.iterrows():
    id=dataset_info["did"]
    if(dataset_info["NumberOfNumericFeatures"]==dataset_info["NumberOfFeatures"]-1):
        f.write(f"name: {dataset_info['name']} id: {dataset_info['did']}\n")
        print(f"name: {dataset_info['name']} id: {dataset_info['did']}")
        
        dataset=openml.datasets.get_dataset(id)
        X,Y,_,_=dataset.get_data(target=dataset.default_target_attribute , dataset_format="dataframe")
        map_dict=encod_dict(Y)
        Y.replace(map_dict, inplace=True)

        X_train, X_test, Y_train, Y_test =train_test_split(X,Y)

        model = RandomForestClassifier(max_depth=15)
        model.fit(X_train,Y_train)
        predictions=model.predict(X_test)
        print(f"Accuracy: {accuracy(Y_test,predictions)}")

        print("")
        print("Spot here for checkpoint each dataset")        
f.close()

