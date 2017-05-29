import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
balance_data = pd.read_csv('train_data.csv',sep= ',', header= None)
balance_data = balance_data.fillna(0)
##X = balance_data.values[:, 2:12]

loan_id = balance_data.values[:,0]
gender = balance_data.replace("M",1).replace("F",2).values[:,1]
marital_status = balance_data.replace("Yes",1).replace("No",2).values[:,2]
dependents = balance_data.replace("3+","3").values[:,3]
qualification = balance_data.replace("Graduate","1").replace("Not Graduate","2").values[:,4]
is_self_employed = balance_data.replace("Yes",1).replace("No",2).values[:,5]
applicant_income = balance_data.values[:,6]
co_applicant_income = balance_data.values[:,7]
loan_amount = balance_data.values[:,8]
loan_amount_term = balance_data.values[:,9]
credit_history = balance_data.values[:,10]
property_area = balance_data.replace("Urban",1).replace("Rural",2).replace("Semiurban",3).values[:,11]

print("Reding the status and appending started")

status_data = pd.read_csv('train_prediction.csv',sep= ',', header= None)
status = status_data.values[:,1]

print("Reding the status and appending is done ")

raw_data1 = {'loan_id': loan_id,
            'status': status,
            "gender":gender,
            "marital_status":marital_status,
            "dependents":dependents,
            "qualification":qualification,
            "is_self_employed":is_self_employed,
            "applicant_income":applicant_income,
            "co_applicant_income":co_applicant_income,
            "loan_amount":loan_amount,
            "loan_amount_term":loan_amount_term,
            "credit_history":credit_history,
            "property_area":property_area            
            }

df1 = pd.DataFrame(raw_data1, columns = ['loan_id', 'status','gender','marital_status','dependents','qualification','is_self_employed','applicant_income'
                                       ,'co_applicant_income','loan_amount','loan_amount_term','credit_history','property_area'])


df1.to_csv('trainData1.csv')




import glob
import csv
myfiles = glob.glob("trainData1.csv")
for file in myfiles:
    lines = open(file).readlines()
    open(file, 'w').writelines(lines[2:])

################Train data prep-processing ends here#############

print("testing data pre processing is started")
import pandas as pd1

test_data = pd1.read_csv('test_data.csv',sep= ',', header= None)
test_data = test_data.fillna(0)

loan_id_t = test_data.values[:,0]
gender_t = test_data.replace("M",1).replace("F",2).values[:,1]
marital_status_t = test_data.replace("Yes",1).replace("No",2).values[:,2]
dependents_t = test_data.replace("3+","3").values[:,3]
qualification_t = test_data.replace("Graduate","1").replace("Not Graduate","2").values[:,4]
is_self_employed_t = test_data.replace("Yes",1).replace("No",2).values[:,5]
applicant_income_t = test_data.values[:,6]
co_applicant_income_t = test_data.values[:,7]
loan_amount_t = test_data.values[:,8]
loan_amount_term_t = test_data.values[:,9]
credit_history_t = test_data.values[:,10]
property_area_t = test_data.replace("Urban",1).replace("Rural",2).replace("Semiurban",3).values[:,11]



raw_data2 = {'loan_id_t': loan_id_t,
            "gender_t":gender_t,
            "marital_status_t":marital_status_t,
            "dependents_t":dependents_t,
            "qualification_t":qualification_t,
            "is_self_employed_t":is_self_employed_t,
            "applicant_income_t":applicant_income_t,
            "co_applicant_income_t":co_applicant_income_t,
            "loan_amount_t":loan_amount_t,
            "loan_amount_term_t":loan_amount_term_t,
            "credit_history_t":credit_history_t,
            "property_area_t":property_area_t            
            }


df_t = pd1.DataFrame(raw_data2, columns = ['loan_id_t','gender_t','marital_status_t','dependents_t','qualification_t','is_self_employed_t','applicant_income_t'
                                       ,'co_applicant_income_t','loan_amount_t','loan_amount_term_t','credit_history_t','property_area_t'])


df_t.to_csv('test_data1.csv')

myfiles = glob.glob("test_data1.csv")
for file in myfiles:
    lines = open(file).readlines()
    open(file, 'w').writelines(lines[2:])


################Test Data preprocessing ends here################




balance_data1 = pd.read_csv('trainData1.csv',sep= ',', header= None)
X = balance_data1.values[:, 3:13]
Y = balance_data1.values[:,2]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3,
                                                     random_state = 100)
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
##print(clf_gini)
clf_gini.fit(X_train, y_train)
balance_outdata = pd.read_csv('test_data1.csv',sep= ',', header= None)
##print(balance_outdata.values[:,0])
##print("predict",clf_gini.predict(balance_outdata.values[:, 2:12]))

##pd.to_csv("newCsv", sep=',', encoding='utf-8')

raw_data = {'loan_id': balance_outdata.values[:,1],
        'status': clf_gini.predict(balance_outdata.values[:, 2:12])}
df = pd.DataFrame(raw_data, columns = ['loan_id', 'status'])


df.to_csv('test_prediction.csv')
df.drop(df.columns[1], axis=1)


y_pred = clf_gini.predict(X_test)
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)

print("#################END###########################")


