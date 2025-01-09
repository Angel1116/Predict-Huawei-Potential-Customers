"""XGBoost with bag-od-encoder"""

import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/嘿統計學習嘿/potential_customers.csv')
data2 = pd.read_csv('/content/drive/MyDrive/嘿統計學習嘿/ad_views_without_engagement.csv')
data['potential_labels'] = 1
data2['potential_labels'] = 0
merged_data = pd.concat([data, data2], ignore_index=True)


"""bag-of-values encoder (recorded by sparse matrix)  """
from scipy.sparse import csr_matrix

def bov_encode_column_sparse(column, separator='^'):

    # 1. filter out non-numeric
    def safe_split_and_convert(x):
        try:
            return [int(val) for val in x.split(separator) if val.isdigit()]
        except:
            return []

    split_col = column.apply(safe_split_and_convert)

    # 2. find all unique values and set index
    unique_values = sorted(set(val for sublist in split_col for val in sublist))
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}

    # 3. record by sparse matrix
    row_indices = []
    col_indices = []
    data_values = []
    for row_idx, sequence in enumerate(split_col):
        for val in sequence:
            if val in value_to_index:
                row_indices.append(row_idx)
                col_indices.append(value_to_index[val])
                data_values.append(1) 

    encoded_sparse_matrix = csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(len(column), len(unique_values))
    )

    column_names = [f'{column.name}_feature_{i}' for i in range(len(unique_values))]
    return encoded_sparse_matrix, column_names


encoded_dataframes = {}
for col in ['u_newsCatInterests', 'u_newsCatDislike', 'u_newsCatInterestsST',
            'u_click_ca2_news', 'i_docId', 'i_s_sourceId', 'i_entities']:
    encoded_sparse_matrix, column_names = bov_encode_column_sparse(merged_data[col])
    encoded_dataframes[col] = (encoded_sparse_matrix, column_names)

for col, (encoded_sparse_matrix, column_names) in encoded_dataframes.items():
    merged_data.drop(columns=[col], inplace=True)
    encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_sparse_matrix, columns=column_names)
    merged_data = pd.concat([merged_data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

print("finish bov encoding")


#------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split

train_data, temp_data = train_test_split(merged_data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.33, random_state=42)

print(f"train: {len(train_data)}")
print(f"valid: {len(valid_data)}")
print(f"test: {len(test_data)}")

#------------------------------------------------------------------------
train_label_counts = train_data['potential_labels'].value_counts(normalize=True)
print("train:")
print(train_label_counts)

valid_label_counts = valid_data['potential_labels'].value_counts(normalize=True)
print("\n valid:")
print(valid_label_counts)

test_label_counts = test_data['potential_labels'].value_counts(normalize=True)
print("\n test:")
print(test_label_counts)

#------------------------------------------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

target = 'potential_labels'

X_train = train_data.drop(columns=['u_userId','potential_labels'])
y_train = train_data[target]
X_valid = valid_data.drop(columns=['u_userId','potential_labels'])
y_valid = valid_data[target]
X_test = test_data.drop(columns=['u_userId','potential_labels'])
y_test = test_data[target]


dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test, label=y_test)



params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}


evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=20000,
    early_stopping_rounds=10,
    evals = [(dtrain, 'train'), (dvalid, 'valid')],
    evals_result=evals_result,
    verbose_eval=10
)

model.save_model('xgboost_model.json')


train_error = evals_result['train']['error']
valid_error = evals_result['valid']['error']
train_accuracy = [1 - error for error in train_error]
valid_accuracy = [1 - error for error in valid_error]

plt.figure(figsize=(10, 6))
plt.plot(train_accuracy, label='Train')
plt.plot(valid_accuracy, label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.show()


y_pred_proba = model.predict(dtest)
y_pred = [1 if x > 0.5 else 0 for x in y_pred_proba]

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
print("\nConfusion Matrix:")
print(cm_df)

importance_dict = model.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': list(importance_dict.keys()),
    'Importance': list(importance_dict.values())
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)