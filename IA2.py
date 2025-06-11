import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Car_Data.csv')
print("******************Analyse des données")
#print(df.head())
#print(df.shape)
#print(df.info())
#Preparations des données
print("*********************Preparation des donnes*****")
df[['quartier', 'region']] = df.Adresse.str.split(',', n = 1, expand = True)
#print(df.head())
df.drop(['region', 'Adresse'], axis = 1, inplace = True)
#print(df.head())
#Modifications des noms des colonnes
df.rename(columns= {'Marque': 'marque', 'Année': 'annee', 'Boite_Vitesse': 'transmission', 'Prix': 'prix', 'Venant_Occasion': 'etat'}, inplace = True)
#print(df.head())
#Reorganiser les variables
print("Apres reorganisation des variables")
df = df[['marque', 'annee', 'transmission', 'prix', 'quartier', 'etat']]
# print(df.head())
#Selections des variables 
liste_marques = df['marque'].unique().tolist()
liste_transmissions = df['transmission'].unique().tolist()
liste_quartiers = df['quartier'].unique().tolist()
joblib.dump(liste_marques, 'marque_list.joblib')
joblib.dump(liste_transmissions, 'transmission_list.joblib')
joblib.dump(liste_quartiers, 'quartier_list.joblib')


cat_data = df.select_dtypes(include = 'object')
quant_data = df.select_dtypes(include = 'number')
# print(cat_data)
# print(quant_data)
# print("Avant suppression des espaces")
# print(np.unique(df.quartier))
# print("Apres la suppression")
df.quartier = df.quartier.str.strip()
# print(np.unique(df.quartier))

#Exploration des varables categorielles
print("*************Exploration des variables categorielles")
#print(cat_data.describe())

#fonctions de countplot
def countplot_func(col):
    val = df[col].value_counts()
    plt.figure(figsize = (25, 3))
    sns.barplot(x= val.index, y = val.values, hue = val.index, palette = sns.hls_palette())
    plt.show()
# countplot de toutes les variables
# for col in cat_data.columns:
#     countplot_func(col)
    
#exploration des données quantitatives
#print(quant_data.describe())
# sns.pairplot(quant_data)
# plt.show()

#fonctions boxplots
def boxplot_func(col):
    plt.figure(figsize=(25, 3))
    sns.boxplot(x = df[col])
    plt.show()
    
#boxplots des toutes les variables quantitatives
# for col in quant_data.columns:
#     boxplot_func(col)
#Preparations des données 
print("*************Preparations des données qualitatives*********************")
from sklearn.preprocessing import LabelEncoder
encoder0 = LabelEncoder()
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
encoder3 = LabelEncoder()
encoders = [encoder0, encoder1, encoder2, encoder3]
for i in range(len(cat_data.columns)):
    encoders[i].fit(df[cat_data.columns[i]])
    df[cat_data.columns[i]] = encoders[i].transform(df[cat_data.columns[i]])
print("***********************Apres Encodage*******************************************")
print(df.head())

for i in range(len(encoders)):
    joblib.dump(encoders[i], cat_data.columns[i]+ '.joblib')

#Preparations des données quantitatives
print("*****************Preparations des données quantitatives************")
def imputation_func(df, col):
    Q1 = df[col].quantile(0.25)
    Q2 = df[col].quantile(0.75)
    IQR = Q2 - Q1
    df[col] = np.where((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q2 + 1.5 * IQR)) , df[col].median(), df[col])

  
for col in ['annee', 'prix']:
    imputation_func(df, col)
    
# for col in ['annee', 'prix']:
#     boxplot_func(col)
    
#Fractionnement et normalisatison
print("**********Fractionnement et normalisations des donnees************")
#Franctionnner les donnees en x variables predicter et y variables cible
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# print(x[:5])
# print(y[:5])

#normalisations des donnees
from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x[:5])

joblib.dump(scaler, 'scaler.joblib')

#Fractionner les donnees en train, val et test
from sklearn.model_selection import train_test_split
x_train, x_vt, y_train, y_vt = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_val, x_test, y_val, y_test = train_test_split(x_vt, y_vt, test_size = 0.5, random_state = 42)
# print("donner d'entrainement")
# print(x_train.shape)
# print(y_train.shape)
# print("donner de validations")
# print(x_val.shape)
# print(y_val.shape)
# print("donner des test")
# print(x_test.shape)
# print(y_test.shape)
#Modelisations 
print("**********Modelisations*****************")
#logistic regressions
print("*****Logistic Regression")
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
#Dictionnaires des paramettres
# params = {"C":np.logspace(-3, 3, 7), "penalty":['l1', 'l2'], 'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']}
# #instantier le model LG
# lg = LogisticRegression()
# #Instantier le model GS
# grid = GridSearchCV(lg, params, cv = 5)
# #Lancer la rechercher
# grid.fit(x_train, y_train)
# Afficher les paramettres optimaux
# print(grid.best_params_)

#Entrainement de LG en utilisant les parametres optimaux
lg = LogisticRegression(C = 10.0, penalty = 'l2', solver = 'saga')

#entrainement 
print("entrainement")
lg.fit(x_train, y_train)
#Evaluations de LG
print("************Evaluations de LG*****************")
#accuracy et F1 score du modeles sur les donnees de valudations
from sklearn.metrics import f1_score, accuracy_score

y_pred_lg = lg.predict(x_val)

acc_lg = accuracy_score(y_val, y_pred_lg)

f1_lg = f1_score(y_val, y_pred_lg)

print(f"Accuracy: {acc_lg: .2f} -- f1 {f1_lg: .2f}")
#accuracy sur les donnees trains
y_pred_train_lg = lg.predict(x_train)

acc_train_lg = accuracy_score(y_train, y_pred_train_lg)
print(f"Accuracy : {acc_train_lg: .2f}")

#varifications grace aux confusionMatricDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cnf_matr = confusion_matrix(y_val, y_pred_lg, labels = [0, 1])

disp_lg = ConfusionMatrixDisplay(confusion_matrix = cnf_matr, display_labels = ['Occasion(0)', 'Venante(1)'])
# disp_lg.plot()
# plt.title("Matrice de confusion")
#plt.show()

#Support Vectors Machines
print("*************SVM********************")
from sklearn.svm import SVC
# params_svm = {"C": np.logspace(-1, 3, 5), 'kernel':['linear', 'rbf']}
# svm = SVC()

# grid_svm = GridSearchCV(svm, params_svm, cv = 5)
# grid_svm.fit(x_train, y_train)
# print(grid_svm.best_params_)

print("*********Entrainement de SVM***************")
svm_model = SVC(C = 10.0, kernel = 'rbf')
#Entrainement
print("Entrainement ")
svm_model.fit(x_train, y_train)
#Evaluations de SVM
print("*********Evaluations de SVM***********")
y_pred_svm = svm_model.predict(x_val)

acc_svm = accuracy_score(y_val, y_pred_svm)
f1_svm = f1_score(y_val, y_pred_svm)

print(f"Accuracy: {acc_svm: .2f} -- F1: {f1_svm: .2f}")

print("*********accuracy sur les variables d'entrainement**********")

y_pred_train_svm = svm_model.predict(x_train)

acc_train_svm = accuracy_score(y_train, y_pred_train_svm)
print(f"Accuracy: {acc_train_svm: .2f}")

print("***Matrix de confusions***********")

cnf_matr_svm = confusion_matrix(y_val, y_pred_svm, labels = [0,1])

disp_svm =ConfusionMatrixDisplay(confusion_matrix = cnf_matr_svm, display_labels = ['Occasion(0)', 'Venante(1)'])
# disp_svm.plot()
# plt.title('Matrice de confusions')
# plt.show()

#Decisions Tree
print("********Decision Tree*************")
from sklearn.tree import DecisionTreeClassifier
params_dt = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15)}
dt = DecisionTreeClassifier()
grid_dt = GridSearchCV(dt, params_dt, cv = 5)
grid_dt.fit(x_train, y_train)
print(grid_dt.best_params_)

#Entrainement du model dt
dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 7)
print("**********Entrainement de DT**********")
dt.fit(x_train, y_train)

#Evaluations de DT
print("Evaluations de DT")
y_pred_dt = dt.predict(x_val)

acc_dt = accuracy_score(y_val, y_pred_dt)
f1_dt = f1_score(y_val, y_pred_dt)
print(f"Accuracy: {acc_dt: .2f} -- F1: {f1_dt: .2f}")
print("Evaluations sur les données trains")
y_pred_train_dt = dt.predict(x_train)
acc_train_dt = accuracy_score(y_train, y_pred_train_dt)
print(f"Accuracy: {acc_train_dt: .2f}")

cnf_matr_dt = confusion_matrix(y_val, y_pred_dt, labels=[0,1])
disp_dt = ConfusionMatrixDisplay(confusion_matrix = cnf_matr_dt, display_labels = ['Occasion(0)', 'Venante(0)'])
#disp_dt.plot()
#plt.title('Matrice de confusion')
#plt.show()

#Random Forest
print("************Rondom Forest*************")
from sklearn.ensemble import RandomForestClassifier
params_rf = param_grid = {
    'n_estimators': list(np.arange(10, 100, 10)),
    'max_depth': list(np.arange(3, 15)),
    'criterion': ['geni', 'entropy']
}

rf = RandomForestClassifier()
# grid_rf = GridSearchCV(estimator = rf, param_grid = params_rf, cv = 5)
# grid_rf.fit(x_train, y_train)

# print(grid_rf.best_params_)
print("entrainement de Rendom Forest")
rf_model = RandomForestClassifier(n_estimators = 60, max_depth = 9, criterion = 'entropy')
#Entrainement 
rf_model.fit(x_train, y_train)

#Evaluations de RF
print("Evaluations de RF")
y_pred_rf = rf_model.predict(x_val)

acc_rf = accuracy_score(y_val, y_pred_rf)
f1_rf = f1_score(y_val, y_pred_rf)
print(f"Accuracy: {acc_rf: .2f} -- F1: {f1_rf: .2f}")
print("Evaluations sur les données trains")
y_pred_train_rf = dt.predict(x_train)
acc_train_rf = accuracy_score(y_train, y_pred_train_rf)
print(f"Accuracy: {acc_train_rf: .2f}")

cnf_matr_rf = confusion_matrix(y_val, y_pred_rf, labels=[0,1])
disp_rf = ConfusionMatrixDisplay(confusion_matrix = cnf_matr_rf, display_labels = ['Occasion(0)', 'Venante(0)'])
# disp_rf.plot()
# plt.title('Matrice de confusion')
# plt.show()

# #Gardient Boosting
print("Gradient Boosting")
from sklearn.ensemble import GradientBoostingClassifier

params_gb = {
    'n_estimators':list(np.arange(10, 100, 10)),
    'max_depth': list(np.arange(3, 15))
}
gb = GradientBoostingClassifier()

grid_gb = GridSearchCV(estimator = gb, param_grid = params_gb, cv = 5)
# grid_gb.fit(x_train, y_train)
# print(grid_gb.best_params_)

gb_model = GradientBoostingClassifier(n_estimators = 80, max_depth = 4)

print("Entrainement ")
gb_model.fit(x_train, y_train)

print("Evaluation de GB model")
y_pred_gb = gb_model.predict(x_val)

acc_gb = accuracy_score(y_val, y_pred_gb)
f1_gb = f1_score(y_val, y_pred_gb)
print(f"Accuracy: {acc_gb: .2f} -- F1: {f1_gb: .2f}")
print("Evaluations sur les données trains")
y_pred_train_gb = gb_model.predict(x_train)
acc_train_gb = accuracy_score(y_train, y_pred_train_gb)
print(f"Accuracy: {acc_train_gb: .2f}")

cnf_matr_gb = confusion_matrix(y_val, y_pred_gb, labels=[0,1])
disp_gb = ConfusionMatrixDisplay(confusion_matrix = cnf_matr_gb, display_labels = ['Occasion(0)', 'Venante(0)'])
# disp_gb.plot()
# plt.title('Matrice de confusion')
# plt.show()

#XGBoost

print("XGBoost")
from xgboost import XGBClassifier
params_xgb = {
     'n_estimaors':list(np.arange(10, 100, 10)),
     'max_depth': list(np.arange(3, 15))
 }

xgb = XGBClassifier()
grid_xgb = GridSearchCV(estimator = xgb, param_grid = params_xgb, cv = 5)
# grid_xgb.fit(x_train, y_train)
# print(grid_xgb.best_params_)
xgb_model = XGBClassifier(n_estimators = 10 , max_depth = 3)
xgb_model.fit(x_train, y_train)

#Evaluations de XGB
print("Evaluations de XGB")
y_pred_xgb = xgb_model.predict(x_val)

acc_xgb = accuracy_score(y_val, y_pred_xgb)
f1_xgb = f1_score(y_val, y_pred_xgb)
print(f"Accuracy: {acc_rf: .2f} -- F1: {f1_rf: .2f}")
print("Evaluations sur les données trains")
y_pred_train_xgb = dt.predict(x_train)
acc_train_xgb = accuracy_score(y_train, y_pred_train_xgb)
print(f"Accuracy: {acc_train_xgb: .2f}")

cnf_matr_xgb = confusion_matrix(y_val, y_pred_xgb, labels=[0,1])
disp_xgb = ConfusionMatrixDisplay(confusion_matrix = cnf_matr_xgb, display_labels = ['Occasion(0)', 'Venante(0)'])
# disp_xgb.plot()
# plt.title('Matrice de confusion')
# plt.show()

#TEST sur le model le plus performant
print("Test sur le model le plus performant donc le GB Model")
x_test_10 = x_test[:10]
y_test_10 = y_test[:10]

y_pred_10 = gb_model.predict(x_test_10)
for i in range(10):
    print(f"La voiture {i + 1} est une voiture {'Venante' if y_pred_10[i] == 1 else 'Occasion'} -- {'Venante' if y_test_10[i] == 1 else 'Occasion'}")
    
# enregistrement du model
joblib.dump(gb_model, 'gb_model.joblib')  

#Deployement avec Gradio
#fonctions de predictions
import gradio as gr
#importer les encoders
encoder0 = joblib.load('marque.joblib')
encoder1 = joblib.load('transmission.joblib')
encoder2 = joblib.load('quartier.joblib')
encoder3 = joblib.load('etat.joblib')

#importer le model
gb_model = joblib.load('gb_model.joblib')
#importer le normaliser
scaler = joblib.load('scaler.joblib')

def pred_fun(marque, annee, transmission, prix, quartier):
    #Encoder les variables marque, transmission, quartier
    marque = encoder0.transform([marque])[0]
    transmission = encoder1.transform([transmission])[0]
    quartier = encoder2.transform([quartier])[0]
    
    #vecteurs des valeurs numerique
    x_new = np.array( (marque, annee, transmission, prix, quartier))
    x_new = x_new.reshape(1, -1)
    #normaliser les données
    x_new = scaler.transform(x_new)
    #predire
    y_pred = gb_model.predict(x_new)
    #arrondir
    y_pred = round(y_pred[0], 2)
    return f"{'venante'if y_pred == 1 else 'occasion'}"
    

def pred_fun_csv(file):
    #lire le fichier csv
    df = pd.read_csv(file)
    predictions = []
    for row in df.iloc[:, :].values:
        new_row = np.array([
            encoder0.transform([row[0]])[0],
            row[1],
            encoder1.transform([row[2]])[0],
            row[3],
            encoder2.transform([row[4]])[0]
            ])
        new_row = new_row.reshape(1, -1)
        new_row = scaler.transform(new_row)
        
        y_pred = gb_model.predict(new_row)
        y_pred = round(y_pred[0], 2)
        predictions.append('venante'if y_pred == 1 else 'occasion')
    df['etat'] = predictions
    df.to_csv('predictions.csv', index = False)
    return 'predictions.csv'
    
    
demo = gr.Blocks(theme = gr.themes.Monochrome())
#creer les inputes
inputs = [
    gr.Dropdown(choices = liste_marques, label = 'marque'),
    gr.Number(label = 'annee'),
    gr.Dropdown(choices = liste_transmissions, label = 'transmission'),
    gr.Number(label = 'prix'),
    gr.Dropdown(choices = liste_quartiers, label = 'quartier')  
]

Outputs = gr.Textbox(label = 'Etat')

interface1 = gr.Interface(
    fn = pred_fun,
    inputs = inputs,
    outputs = Outputs,
    title = "Saisir les donnees",
    description = """Cette modele predi si une voiture est une voiture d'occasion ou bien si elle est Venante a partir de quelques informations"""
)

interface2 = gr.Interface(
    fn = pred_fun_csv,
    inputs = gr.File(label = 'Televerser le fichier csv'),
    outputs = gr.File(label = 'Telecharger le ficher csv'),
    title = "Televerser un fichier csv",
    description = """Cette modele predi si une voiture est une voiture d'occasion ou bien si elle est Venante a partir de quelques informations"""
)

with demo:
    gr.TabbedInterface([interface1, interface2], ['simple prediction', 'multiple predictions'])
    
demo.launch()