# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import math
import seaborn as sn
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')
test = pd.read_csv('test.csv')

# %%
train.isnull().sum()

# %%
# Separate features and labels
X_train = train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_train = train[['label_1', 'label_2', 'label_3', 'label_4']]
X_val = valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_val = valid[['label_1', 'label_2', 'label_3', 'label_4']]
X_test = test.drop(['ID'], axis=1)

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ### Label_1

# %%
plt.figure(figsize=(18, 6))
sn.countplot(data=y_train, x='label_1', color='teal')
plt.xlabel('Speaker', fontsize=12)

# %%
len(y_train['label_1'].unique())

# %%
from sklearn.metrics import classification_report

def get_score(model, X_train, y_train, X_val, y_val, verbose = False):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    if verbose:
        print(classification_report(y_val, y_pred))

    return accuracy_score(y_val, y_pred)

# %%
def evaluate_models(models, X_train, y_train, X_valid, y_valid):
    model_accuracies = {
        model_name: get_score(model, X_train, y_train, X_valid, y_valid)
        for model_name, model in models.items()
    }
    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(model_accuracies.keys(), model_accuracies.values())
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.show()


# %%
acc_logistic = cross_val_score(LogisticRegression(), X_train, y_train['label_1'], cv=3)

# %%
acc_svc = cross_val_score(SVC(C=1000, gamma=0.001), X_train, y_train['label_1'], cv=3).mean()

# %%
acc_svc

# %%
acc_rfc = cross_val_score(RandomForestClassifier(), X_train, y_train['label_1'], cv=3)

# %%
print("Logistic Regression Accuracy: %0.2f (+/- %0.2f)" % (acc_logistic.mean(), acc_logistic.std() * 2))
print("SVC Accuracy: %0.2f (+/- %0.2f)" % (acc_svc.mean(), acc_svc.std() * 2))
print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (acc_rfc.mean(), acc_rfc.std() * 2))

# %%
cross_val_score(LogisticRegression(), X_train_scaled, y_train['label_1'], cv=3).mean()

# %%
get_score(
    LogisticRegression(), 
    X_train_scaled, y_train['label_1'], 
    X_val_scaled, y_val['label_1']
)

# %%
pca = PCA(n_components=0.95, svd_solver = 'full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
models = {
    'SVM': SVC(C=1000, gamma=0.001),
    'LogisticRegression': LogisticRegression(),
    'KNN':KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(),
}

evaluate_models(models, X_train_pca, y_train['label_1'], X_val_pca, y_val['label_1'])  

# %%
from xgboost import XGBClassifier

get_score(
    XGBClassifier(num_class=len(y_train['label_1'].unique()), tree_method='gpu_hist', gpu_id= 0),
    X_train_pca, y_train['label_1']-1,
    X_val_pca, y_val['label_1']-1
)

# %%
get_score(
    SVC(C=1000, gamma=0.001, kernel='rbf'),
    X_train_pca, y_train['label_1'],
    X_val_pca, y_val['label_1'],
    verbose=True
)

# %%
cross_val_score(SVC(C=1000, gamma=0.001, kernel='rbf'), X_train_pca, y_train['label_1'], cv=3).mean()

# %%
# Hyper Parameter Tuning For SVC

from sklearn.model_selection import GridSearchCV
  
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(X_train_pca, y_train['label_1'])
grid.best_params_

# %%
# grid.best_estimator_

# %%
# Saving the model

# import joblib
# loaded_model = joblib.load('svc_label_1.pkl')

# loaded_model

# %%
get_score(
    CatBoostClassifier(loss_function='MultiClass', task_type="GPU",
                           devices='0:1'),
    X_train_pca, y_train['label_1'],
    X_val_pca, y_val['label_1']
)

# %%
best_model_label_1 = SVC(C=1000, gamma=0.001, kernel='rbf')

# %%

pred_label1 = best_model_label_1.fit(X_train_pca, y_train['label_1']).predict(X_test_pca)

# %%
pred_label1.shape

# %%
get_score(
    SVC(C=1000, gamma=0.001, kernel='rbf'),
    X_train_pca, y_train['label_1'],
    X_val_pca, y_val['label_1'],
    verbose= True
)

# %%
y_pred = SVC(C=1000, gamma=0.001, kernel='rbf').fit(X_train_pca, y_train['label_1']).predict(X_val_pca)
confusion = metrics.confusion_matrix(y_val['label_1'], y_pred)

# %%
plt.figure(figsize=(20, 10))
sn.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# %%
best_model_label_1 = SVC(C=1000, gamma=0.001, kernel='rbf', probability=True)
best_model_label_1.fit(X_train_pca, y_train['label_1'])
pred = best_model_label_1.predict(X_val_pca)
accuracy_score(y_val['label_1'], pred )

# %%
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=y_train['label_1'].unique())

# %% [markdown]
# ### Label_2

# %%
train['label_2'].isnull().sum()

# %%
label2_train = train.copy()
label2_valid = valid.copy()
label2_test = test.copy()

# %%
label2_train = label2_train.dropna(subset=['label_2'])
label2_valid = label2_valid.dropna(subset=['label_2'])

# %%
label2_train['label_2'].isnull().sum()

# %%
X_train = label2_train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_train = label2_train[['label_1', 'label_2', 'label_3', 'label_4']]
X_val = label2_valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_val = label2_valid[['label_1', 'label_2', 'label_3', 'label_4']]
X_test = label2_test.drop(['ID'], axis=1)

# %%
plt.figure(figsize=(18, 6))
ax = sn.histplot(data=y_train, x='label_2', bins=20, kde=False)
plt.xlabel('Speaker Age')

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

plt.show()

# %%
y_train['label_2'].nunique()

# %%
y_train['label_2'].value_counts()

# %%
X_train.shape

# %%
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# %%
pca = PCA(n_components=0.95, svd_solver = 'full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
from sklearn.model_selection import GridSearchCV

leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Use GridSearch
clf = GridSearchCV(KNeighborsClassifier(), hyperparameters, cv=10)
best_model = clf.fit(X_train_pca,y_train['label_2'])
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

best_model.best_estimator_

# %%
get_score(
    KNeighborsClassifier(n_neighbors=11),
    X_train_pca, y_train['label_2'],
    X_val_pca, y_val['label_2']
)

# %%
# label2_train_filtered = label2_train[label2_train['label_2'] < 45]
# X_train_filtered = label2_train_filtered.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
# y_train_filtered = label2_train_filtered[['label_1', 'label_2', 'label_3', 'label_4']]

# %%
# X_train_filtered.shape

# %%
# scaler = StandardScaler()
# X_train_scaled_filtered = scaler.fit_transform(X_train_filtered)
# X_val_scaled_filtered = scaler.transform(X_val)
# X_test_scaled_filtered = scaler.transform(X_test)

# pca = PCA(n_components=0.95, svd_solver = 'full')
# X_train_pca_filtered = pca.fit_transform(X_train_scaled_filtered)
# X_val_pca_filtered = pca.transform(X_val_scaled_filtered)
# X_test_pca_filtered = pca.transform(X_test_scaled_filtered)


# %%
# from sklearn.model_selection import GridSearchCV
  
# # defining parameter range
# param_grid = {'C': [1000, 1200, 1300, 1500, 1600, 2000, 1800, 2200],
#               'kernel': ['rbf']} 
  
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# # fitting the model for grid search
# grid.fit(X_train_pca, y_train['label_2'])
# grid.best_params_

# %%
get_score(
    SVC(C=1000),
    X_train_pca, y_train['label_2'],
    X_val_pca, y_val['label_2']
)

# %%
get_score(
    CatBoostClassifier(loss_function='MultiClass'
                       , learning_rate = 0.15),
    X_train_pca, y_train['label_2'],
    X_val_pca, y_val['label_2']
) * 100

# %%
cross_val_score(SVC(C=1000), X_train_pca, y_train['label_2'], cv=5).mean()

# %%
best_model_label_2 = SVC(C=1000)
pred_label2 = best_model_label_2.fit(X_train_pca, y_train['label_2']).predict(X_test_pca)

# %% [markdown]
# ### Label 3

# %%
X_train = train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_train = train[['label_1', 'label_2', 'label_3', 'label_4']]
X_val = valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_val = valid[['label_1', 'label_2', 'label_3', 'label_4']]
X_test = label2_test.drop(['ID'], axis=1)

# %%
ax = sn.countplot(x=y_train['label_3'])

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, color='black')
    
plt.xlabel('Speaker Gender')

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# %%
pca = PCA(n_components=0.95, svd_solver = 'full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
get_score(
    SVC(),
    X_train_pca,y_train['label_3'],
    X_val_pca,y_val['label_3'],
)

# %%
cross_val_score(SVC(), X_train_pca, y_train['label_3'], cv=5).mean()

# %%
best_model_label_3 = SVC()
pred_label3 = best_model_label_3.fit(X_train_pca, y_train['label_3']).predict(X_test_pca)

# %% [markdown]
# ### Label 4

# %%
ax = sn.countplot(x=y_train['label_4'])

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, color='black')
    
plt.xlabel('Speaker Gender')

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# %%
pca = PCA(n_components=0.95, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
get_score(
    SVC(class_weight='balanced', C=1000),
    X_train_pca,y_train['label_4'],
    X_val_pca,y_val['label_4']
)

# %%
cross_val_score(SVC(class_weight='balanced', C=1000), X_train_pca, y_train['label_4'], cv=5, scoring='accuracy').mean()

# %%
best_model_label_4 = SVC(class_weight='balanced', C=1000)
pred_label4 = best_model_label_4.fit(X_train_pca, y_train['label_4']).predict(X_test_pca)

# %% [markdown]
# ### Generating Output

# %%
output_df = test[['ID']]
output_df['label_1'] = pred_label1
output_df['label_2'] = pred_label2
output_df['label_3'] = pred_label3
output_df['label_4'] = pred_label4

# %%
output_df.head()

# %%
output_df.to_csv('outputs/output_layer7.csv', index=False)

# %%
output_df.shape


