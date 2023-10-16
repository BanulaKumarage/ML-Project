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
from sklearn.pipeline import Pipeline

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
train = pd.read_csv('layer7/train.csv')
valid = pd.read_csv('layer7/valid.csv')
test = pd.read_csv('layer7/test.csv')

# %%
# Separate features and labels
X_train = train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_train = train[['label_1', 'label_2', 'label_3', 'label_4']]
X_val = valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_val = valid[['label_1', 'label_2', 'label_3', 'label_4']]
X_test = test.drop(['ID'], axis=1)

# %%
scaler = StandardScaler()
pca = PCA(n_components=0.95, svd_solver = 'full')
model = SVC(C=1000, gamma=0.001, kernel='rbf')

pipeline = Pipeline([
    ('scaler', scaler),
    ('pca', pca),
    ('classifier', model)
])

pipeline.fit(X_train, y_train['label_1'])

# %%
pipeline.score(X_val, y_val['label_1'])

# %%
model = SVC(C=1000, gamma=0.001, kernel='rbf')
model.fit(X_train, y_train['label_1'])

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95, svd_solver = 'full')
X_train_pca = pd.DataFrame(pca.fit_transform(X_train_scaled))
X_val_pca = pd.DataFrame(pca.transform(X_val_scaled))
X_test_pca = pd.DataFrame(pca.transform(X_test_scaled))

# %%
svm = SVC(C=1000, gamma=0.001, kernel='rbf', probability=True)
svm.fit(X_train_pca, y_train['label_1'])

# %%
svm.score(X_val_pca, y_val['label_1'])

# %%
explainer = shap.KernelExplainer(svm.predict, X_val_pca)
shap_values = explainer(X_val_pca)

# %%
class_names = list(
    map(
        lambda x: f'Class_{str(x)}',
        set(sorted(y_train['label_1'].unique().tolist())),
    )
)

feature_names = list(map(
    lambda x: f'Feature_{str(x)}',
    X_train_pca.columns,
))

# %%
#Explain using Lime tabuler explainer
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X_train_pca.values, feature_names=feature_names, class_names=class_names, discretize_continuous=True)
exp = explainer.explain_instance(X_val_pca.values[0], svm.predict_proba, num_features=10)
exp.show_in_notebook(show_table=True, show_all=False)

# %%
exp.save_to_file('lime.html')

# %%
xgb_classifier = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
xgb_classifier.fit(np.array(X_train_pca), y_train['label_1'])

# %%
xgb_classifier.score(np.array(X_val_pca), y_val['label_1'])

# %%
xgb.plot_importance(xgb_classifier, importance_type='weight',max_num_features=10)  # You can use 'weight', 'gain', or 'cover'
plt.show()


# %%
# Explain XGB using Lime tabuler explainer
import lime
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_pca.values, feature_names=feature_names, class_names=class_names, discretize_continuous=True)
exp = explainer.explain_instance(X_val_pca.values[0], xgb_classifier.predict_proba, num_features=10)

# %%
exp

# %%
import shap
shap.initjs()

# %%
explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(X_val_pca.values)

# %%
import matplotlib.pyplot as plt
shap.summary_plot(
    shap_values, X_val_pca.values, plot_type="bar", 
    class_names= class_names, feature_names = feature_names, show=False, plot_size=(30, 6))
plt.legend(prop={'size': 6},bbox_to_anchor=(1.1, 0.9),ncol=2)
plt.savefig("summary_plot_label1.png", bbox_inches='tight')
plt.show()

# %%
shap.plots.force(explainer.expected_value[0], shap_values[0][0], feature_names=feature_names)


