# %%Kesifsel Veri Analizi EDA

import numpy as np #linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # visualization
import seaborn as sns # visualization
import plotly.express as px # visualization

import missingno as msno # missing values analysisis

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV # model selection
from sklearn.model_selection import train_test_split # splitting data
from sklearn.tree import DecisionTreeClassifier # decision tree model
from sklearn.metrics import precision_score,confusion_matrix # model evaluation
from sklearn.ensemble import RandomForestClassifier # random forest model 

from sklearn import tree

df = pd.read_csv("water_potability.csv ")

described=df.describe()

df.info()


#dependent variable analysis (bagimli degisken analizi)
d = df["Potability"].value_counts().reset_index()
d.columns = ["Potability", "count"]

fig = px.pie(
    d,
    values="count",
    names="Potability",
    hole=0.35,
    opacity=0.8,
    labels={"Potability": "Potability", "count": "Number of Samples"},
)
fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
fig.update_traces(textposition="outside", textinfo="percent+label")
fig.show()

fig.write_html("potability_pie_chart.html")



# korelasyon analizi
sns.clustermap (df.corr(), cmap = "vlag", dendrogram_ratio= (0.1,0.2), annot = True, linewidths=0.8, figsize =(10,10))
plt.show()

# distribution of features
non_potable = df.query("Potability == 0")
potable = df.query("Potability == 1")


plt.figure()
for ax, col in enumerate(df.columns[:9]):
    plt.subplot(3,3, ax + 1)
    plt.title(col)
    sns.kdeplot(x = non_potable [col], label = "Non Potable")
    sns.kdeplot(x = potable [col], label = "Potable")
    plt.legend()

plt.tight_layout()

#missing values analysis
msno.matrix(df)
plt.show()


# %% Preprocessing: missing value problem, train-test split, normalization
print(df.isnull().sum())

df["ph"] = df["ph"].fillna(df["ph"].mean())
df["Sulfate"] = df["Sulfate"].fillna(df["Sulfate"].mean())
df["Trihalomethanes"] = df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean())

print(df.isnull().sum())

# Train-test split
X = df.drop("Potability", axis=1).values
y = df["Potability"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Min-max normalization
x_train_max = np.max(X_train)
x_train_min = np.min(X_train)

X_train = (X_train - x_train_min) / (x_train_max - x_train_min)
X_test = (X_test - x_train_min) / (x_train_max - x_train_min)


# %% Modelling: decision tree and random forest
models = [("DTC", DecisionTreeClassifier (max_depth=5)),
           ("RF", RandomForestClassifier())]

finalResult = [] # score list
cmList = [] # confusion matrix list
for name, model in models:

    model.fit(X_train, y_train) # training
    
    model_result = model.predict(X_test) # prediction

    score = precision_score (y_test, model_result)
    finalResult.append((name, score))

    cm = confusion_matrix(y_test, model_result)
    cmList.append((name, cm))

print (finalResult)
for name, i in cmList:
    plt.figure()
    sns.heatmap(i, annot = True, linewidths=0.8, fmt=".0f")
    plt.title(name)
    plt.show()


#%% Evaluation: decision tree visualization
dt_clf = models[0][1]

plt.figure(figsize = (25,20))
tree.plot_tree(dt_clf, feature_names= df.columns.tolist()[:-1],
                class_names = ["0","1"],
                filled = True,
                precision = 5)
plt.show()

#%%  Hyperparameter tuning: random forest
model_params = {
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "n_estimators": [10, 50, 100],
            "max_features": ["sqrt", "log2", None],
            "max_depth": list(range(1, 21, 3))
        }
    }
}


cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
scores = []

for model_name, params in model_params.items():

    rs = RandomizedSearchCV(params["model"], params["params"], cv=cv, n_iter=10) 
    rs.fit(X,y)
    scores.append([model_name, dict(rs.best_params_), rs.best_score_])

print(scores)
results_df = pd.DataFrame(finalResult, columns=["Model", "Precision"])
print(results_df)
