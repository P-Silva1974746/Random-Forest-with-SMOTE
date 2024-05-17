from random_forest import RandomForestClassifier
from metrics import accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import openml
from openml import tasks
from SMOTE import smote
import torch

def encod_dict(X):
    dic={}
    values=X.unique()
    num=0
    for value in values:
        dic.update({value: num})
        num+=1
    
    return dic

# def get_classes(predictions, map_dict):
# still to implement


# benchmark=openml.study.get_suite(suite_id=99)
# print('''Getting datasets ...
# ''')
# datasets_info_df=tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
# f = open("datasets_used.txt","w")

#Criar uma instância do SMOTE com os parâmetros desejados
smote_i = smote(distance='euclidian', dims=4, k=5)# dimns is number of atributes that are not the target

# for index, dataset_info in datasets_info_df.iterrows():
#     id=dataset_info["did"]
#     if(dataset_info["NumberOfNumericFeatures"]==dataset_info["NumberOfFeatures"]-1):
#         f.write(f"name: {dataset_info['name']} id: {dataset_info['did']}\n")
#         print(f"name: {dataset_info['name']} id: {dataset_info['did']}")
        
#         dataset=openml.datasets.get_dataset(id)
#         X,Y,_,_=dataset.get_data(target=dataset.default_target_attribute , dataset_format="dataframe")
#         map_dict=encod_dict(Y)
#         Y.replace(map_dict, inplace=True)
        
#         # Aplicar o SMOTE aos conjuntos de dados de treinamento
        
#         X_train, X_test, Y_train, Y_test =train_test_split(X,Y)

#         X_train= torch.tensor(X_train.values)
#         Y_train= torch.tensor(Y_train.values)

#         X_train, Y_train = smote_i.fit_generate(X_train, Y_train)


#         model = RandomForestClassifier(max_depth=15)
#         model.fit(X_train,Y_train)
#         predictions=model.predict(X_test)

#         print(f"Accuracy: {accuracy(Y_test,predictions)}")

#         print("")
#         print("Spot here for checkpoint each dataset")        
# f.close()


dataset=openml.datasets.get_dataset(11)
X,Y,_,_=dataset.get_data(target=dataset.default_target_attribute , dataset_format="dataframe")
map_dict=encod_dict(Y)
Y.replace(map_dict, inplace=True)
# Aplicar o SMOTE aos conjuntos de dados de treinamento
        
X_train, X_test, Y_train, Y_test =train_test_split(X,Y)


print(f"dataset antes smote {X_train.shape}")
model = RandomForestClassifier(max_depth=15, smote=smote_i , smote_type="binary")
model.fit(X_train,Y_train)
predictions=model.predict(X_test)
print(predictions)
print(type(predictions))
#print(roc_auc_score(Y_test,predictions))
print(f"Accuracy: {accuracy(Y_test,predictions)}")


