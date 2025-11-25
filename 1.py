# 1) Divide data into training and test part, Compare
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("heart_attack.csv")

y = data['HeartDiseaseorAttack'].values
X = data.drop('HeartDiseaseorAttack', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

knn = KNeighborsClassifier()
nb = GaussianNB()

knn.fit(X_train, y_train)
nb.fit(X_train, y_train)

print("KNN train score =", knn.score(X_train, y_train))
print("KNN test score  =", knn.score(X_test, y_test))

print("GaussianNB train score =", nb.score(X_train, y_train))
print("GaussianNB test score  =", nb.score(X_test, y_test))
#--------------------------------------------------------------------------
# 2) GridSearchCV
#best score and best parameters
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


data = pd.read_csv("heart_attack.csv")

y = data['HeartDiseaseorAttack'].values
X = data.drop('HeartDiseaseorAttack', axis=1).values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = KNeighborsClassifier()

parameters = dict()
parameters["n_neighbors"] = [3, 5, 7, 9, 11, 13]

search = GridSearchCV(estimator=model, param_grid=parameters, scoring="accuracy", cv=5)

print(search.fit(x_train, y_train).score(x_test, y_test))
print(search.best_params_)
#---------------------------------------------------------------------
# 3) cluster, KMeans
import pandas as pd
import matplotlib.pyplot as plt #hart
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("wine.csv")

plt.scatter(data['X1'], data['X2'])
plt.xlabel("X1")
plt.ylabel("X2")

model = KMeans(n_clusters=3)
labels = model.fit_predict(data.values)

print(labels)

plt.scatter(data['X1'], data['X2'], c=labels)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], color='black', s=100)

overall_center = model.cluster_centers_.mean(axis=0)

plt.scatter(overall_center[0], overall_center[1],
            color='red', s=100, marker="X", label="Overall center")

#Silhouette score
X = data.values
labels = KMeans(3).fit_predict(X)
print("Silhouette score =", silhouette_score(X, labels))
#----------------------------------------------------------------

plt.xlabel("X1")
plt.ylabel("X2")
plt.show() #chart showing
#---------------------------------------------------------------------------------
# 4) LinearRegression
import pandas as pd
from sklearn.linear_model import LinearRegression #LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("auto.csv")

data['car_length'] = data['car name'].str.len()
data = data.drop('car name', axis=1)

y = data['mpg'].values
X = data.drop('mpg', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

print("Train score =", model.score(X_train, y_train))
print("Test score  =", model.score(X_test, y_test))
#-------------------------------------------------------------------------------------
# 5) MinMaxScaler, PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

data = pd.read_csv("auto.csv")

data['car_length'] = data['car name'].str.len()
data = data.drop('car name', axis=1)

y = data['mpg'].values
X = data.drop('mpg', axis=1).values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=2)
X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = Ridge()
model.fit(X_train, y_train)

print("Ridge + PCA score =", model.score(X_test, y_test))
#-------------------------------------------------------------------------------------------
# 6) LogisticRegression, print accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("exam.csv")

y = data["Passed"].values
X = data.drop("Passed", axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Accuracy =", model.score(X_test, y_test))
#---------------------------------------------------------------------------
# 7) ROC-curve. print AUC
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv("clients.csv")

y = data["Default"].values
X = data.drop("Default", axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
score = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

print("AUC =", score)
#-------------------------------------------------------------------------------------
# 8)Pipelines
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

data = pd.read_csv("students.csv")

y = data["GPA"].values
X = data.drop("GPA", axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

pipe = Pipeline([
    ("scaler", MinMaxScaler()),
    ("pca", PCA(n_components=2)),
    ("model", LinearRegression())
])

pipe.fit(X_train, y_train)

print("Pipeline score =", pipe.score(X_test, y_test))
#---------------------------------------------------------------------------------
# 9) Dendrogram
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_csv("fruits.csv")

Z = linkage(data.values, method="ward")

plt.figure(figsize=(8,5))
dendrogram(Z)
plt.show()

