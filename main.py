
#  install matplotlib/sklearn(scikit)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

iris = pd.read_csv(r"C:\Users\Yami\PycharmProjects\Iris\Iris.csv")
features = iris.drop(['Species', 'Id'], axis=1)

target = iris['Species']

species_colors = {
    'Iris-setosa': 'purple',
    'Iris-versicolor': 'orange',
    'Iris-virginica': 'blue'
}
colors = target.map(species_colors)
#le = LabelEncoder()
#y_encoded = le.fit_transform(target)

# splitting dataset into training and test
X_training_data, X_test_data, y_training_target, y_test_target = train_test_split(features, target, random_state=0)
X_training_data2, X_test_data2, y_training_target2, y_test_target2 = train_test_split(features, target, random_state=0
                                                                                      ,test_size=0.1)
X_training_data3, X_test_data3, y_training_target3, y_test_target3 = train_test_split(features, target, random_state=0
                                                                                      ,test_size=0.4)
X_training_data4, X_test_data4, y_training_target4, y_test_target4 = train_test_split(features, target, random_state=0
                                                                                      ,test_size=0.7)
knn = KNeighborsClassifier(n_neighbors=1)
knn2 = KNeighborsClassifier(n_neighbors=1)
knn3 = KNeighborsClassifier(n_neighbors=1)
knn4 = KNeighborsClassifier()

# regardless of the n_neighbors value 1-9 the prediction score stays the same (??)

# KNN
knn.fit(X_training_data, y_training_target)
test_predict = knn.predict(X_test_data)
print("KNN Classification Report:")
print(metrics.classification_report(y_test_target, test_predict, digits=3))

#KNN TestSize 10%
knn2.fit(X_training_data2, y_training_target2)
test_predict2 = knn.predict(X_test_data2)
print("KNN Classification Report (test size 10%):")
print(metrics.classification_report(y_test_target2, test_predict2, digits=3))

#KNN TestSize 40%
knn3.fit(X_training_data3, y_training_target3)
test_predict3 = knn.predict(X_test_data3)
print("KNN Classification Report (test size 40%):")
print(metrics.classification_report(y_test_target3, test_predict3, digits=3))

#KNN TestSize 70%
knn3.fit(X_training_data4, y_training_target4)
test_predict4 = knn.predict(X_test_data4)
print("KNN Classification Report (test size 70%):")
print(metrics.classification_report(y_test_target4, test_predict4, digits=3))



# SVM

svm = SVC(kernel='linear', C=1.0)
svm.fit(X_training_data, y_training_target)
feature_importance_svm = abs(svm.coef_[0])

test_predict_svm = svm.predict(X_test_data)  # Predict on the test data

# SVM GRID SEARCH
# Grid search is exhaustive - searches every possibility within the dist
# good for small dist or when you need certainity
param_grid_svm = {
    'C': [0.1, 0.5, 1, 3, 5, 6, 7, 8, 9, 10, 14],
    'kernel': ['linear', 'rbf', 'poly'], #rbf = Radial Basic Function
    'gamma': [0.01, 0.5, 0.1, 1, 3, 5]
}
svm2 = SVC()
grid_search = GridSearchCV(svm2, param_grid_svm, cv=5, scoring='accuracy')
grid_search.fit(X_training_data, y_training_target)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("SVM GRID SEARCH OPTIMIZATION")
print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_score) #optimistic score as it bases it solely on the training data

# Training new SVM with optimal parameters
best_svm = SVC(**best_params)
best_svm.fit(X_training_data, y_training_target)
test_predict_best_svm = best_svm.predict(X_test_data)
print("Optimized SVM")
print(metrics.classification_report(y_test_target, test_predict_best_svm, digits=3))


# SVM RandomSearch
# Random search picks random samples from given dist it doesn't test ALL of them, might vary between runs because of the rng model,
# good for large dist

svm3 = SVC()
random_search_svm = RandomizedSearchCV(svm3,param_distributions=param_grid_svm,n_iter=10,cv=5,scoring='accuracy',random_state=0)
random_search_svm.fit(X_training_data,y_training_target)
best_params_rng_svm = random_search_svm.best_params_
best_score_rng_svm = random_search_svm.best_score_
print("SVM RANDOM SEARCH OPTIMIZATION")
print("Best Hyperparameters:", best_params_rng_svm)
print("Best Accuracy:", best_score_rng_svm)

best_svm2 = SVC(**best_params_rng_svm)
best_svm2.fit(X_training_data,y_training_target)

test_predict_rng_best_svm = best_svm2.predict(X_test_data)
print("Optimized RNG SVM Classification Report:")
print(metrics.classification_report(y_test_target, test_predict_rng_best_svm, digits=3))

#KNN Optimized (grid search)

param_grid_knn = {
    'n_neighbors' : np.arange(1,20), # Number of neighbors
    'weights' : ['uniform','distance'], # Weight function used in prediction
    'p' : [1,2] # Power parameter for the Minkowski distance
}
knn2 = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn2,param_grid_knn,cv=5,scoring='accuracy')
grid_search_knn.fit(X_training_data,y_training_target)
best_params_knn = grid_search_knn.best_params_
best_score_knn = grid_search_knn.best_score_
print("Best Hyperparameters:", best_params_knn)
print("Best Accuracy:", best_score_knn)
best_knn = KNeighborsClassifier(**best_params_knn)
best_knn.fit(X_training_data,y_training_target)
test_predict_knn = best_knn.predict(X_test_data)
print("Optimized KNN")
print(metrics.classification_report(y_test_target,test_predict,digits=3))




#Pairplot

sns.pairplot(iris, diag_kind='hist', hue="Species", palette=species_colors,
             vars=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
plt.show()
plt.figure(figsize=(8, 6))
# Feature importance by random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_training_data, y_training_target)
feature_importance_rf = random_forest.feature_importances_
feature_names = X_training_data.columns
sns.barplot(y=feature_importance_rf, x=feature_names)
plt.title("Feature importance (Random Forest)")

plt.show()
# Feature importances by SVM
plt.bar(range(len(feature_importance_svm)), feature_importance_svm)
plt.xticks(range(len(feature_importance_svm)), feature_names, rotation=45)
plt.xlabel('Feature')
plt.ylabel('Absolute SVM Coefficient')
plt.title('Feature Importances (SVM with Linear Kernel)')
#plt.show()

# score = accuracy
# print(knn.score(X_test_data,y_test_target))

print("SVM Classification Report:")
print(metrics.classification_report(y_test_target, test_predict_svm, digits=3))

#Cross Valuation

scores = cross_val_score(svm,features,target,cv=5) # table with scores for [5] different tests of the SVM model
print("Accuracy of each fold:")
for score in scores:
    print("{:.3f}".format(score))
print("Overall %0.3f accuracy\nStandard deviation %0.3f" % (scores.mean(), scores.std()))
