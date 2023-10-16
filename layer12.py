# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import pickle

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
from sklearn.metrics import classification_report

def get_score(model, X_train, y_train, X_val, y_val, verbose = False):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    if verbose:
        print(classification_report(y_val, y_pred))

    return accuracy_score(y_val, y_pred) * 100

# %%
train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')
test = pd.read_csv('test.csv')

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

# %%
pca = PCA(n_components=0.95, svd_solver = 'full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %% [markdown]
# ### Label_1

# %%
best_model_layer1 = SVC(C=1500, gamma=0.001, kernel='rbf')

# %%
best_model_layer1.fit(X_train_pca, y_train['label_1'])
pred_label1 = best_model_layer1.predict(X_test_pca)

# %%
best_model_layer1.score(X_val_pca, y_val['label_1'])

# %%
with open('best_model_layer1.pkl', 'rb') as file:  
    model = pickle.load(file)
pred_label1 = model.predict(X_val_pca)

# %%
with open('best_model_layer1.pkl', 'wb') as file:
    pickle.dump(best_model_layer1, file)

# %%
# cross_val_score(best_model_layer1, X_train_scaled, y_train['label_1'], cv=2, scoring='accuracy').mean()

# %% [markdown]
# ### Label_3

# %%
best_model_label_3 = SVC(C=100, gamma=0.001, kernel='rbf')

# %%
best_model_label_3.fit(X_train_pca, y_train['label_3'])
pred_label3 = best_model_label_3.predict(X_test_pca)

# %%
best_model_label_3.score(X_val_pca, y_val['label_3'])

# %%
with open('best_model_label_3.pkl', 'wb') as file:
    pickle.dump(best_model_label_3, file)

# %% [markdown]
# ### Label_4

# %%
best_model_label_4 = SVC(C=1000, gamma='auto', class_weight='balanced')
best_model_label_4.fit(X_train_pca, y_train['label_4'])
pred_labe4 = best_model_label_4.predict(X_test_pca)

# %%
with open('best_model_label_4.pkl', 'wb') as file:
    pickle.dump(best_model_label_4, file)

# %%
best_model_label_4.score(X_val_pca, y_val['label_4'])

# %% [markdown]
# ### Label_2

# %%
label2_train = train.copy()
label2_valid = valid.copy()
label2_test = test.copy()

# %%
label2_train = label2_train.dropna(subset=['label_2'])
label2_valid = label2_valid.dropna(subset=['label_2'])

# %%
X_train = label2_train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_train = label2_train[['label_1', 'label_2', 'label_3', 'label_4']]
X_val = label2_valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_val = label2_valid[['label_1', 'label_2', 'label_3', 'label_4']]
X_test = label2_test.drop(['ID'], axis=1)

# %%
from sklearn.preprocessing import RobustScaler


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
best_model_label_2 = SVC(C=100, gamma=0.001, kernel='rbf')
best_model_label_2.fit(X_train_pca, y_train['label_2'])
pred_labe2 = best_model_label_2.predict(X_test_pca)

# %%
best_model_label_4.score(X_val_pca, y_val['label_2'])

# %%
with open('best_model_label_2.pkl', 'wb') as file:
    pickle.dump(best_model_label_2, file)

# %%
output_df = test[['ID']]
output_df['label_1'] = pred_label1
output_df['label_2'] = pred_labe2
output_df['label_3'] = pred_label3
output_df['label_4'] = pred_labe4

# %%
output_df.to_csv('output/output_layer12.csv', index=False)

# %%
output_df.head()

# %%



