"""CatBoost with bag-od-encoder"""

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('/content/drive/MyDrive/potential_customers.csv')
data2 = pd.read_csv('/content/drive/MyDrive/ad_views_without_engagement.csv')
data['potential_labels'] = 1
data2['potential_labels'] = 0
merged_data = pd.concat([data, data2], ignore_index=True)
print(merged_data.head())

train_data, temp_data = train_test_split(merged_data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.33, random_state=42)

print(f"train: {len(train_data)}")
print(f"valid: {len(valid_data)}")
print(f"test: {len(test_data)}")


"""bov encoding (recorded by sparse matrix)  """
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

print(merged_data)

import pandas as pd
from sklearn.model_selection import train_test_split

train_data, temp_data = train_test_split(merged_data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.33, random_state=42)


print(f"train: {len(train_data)}")
print(f"valid: {len(valid_data)}")
print(f"test: {len(test_data)}")

train_label_counts = train_data['potential_labels'].value_counts(normalize=True)
print("train:")
print(train_label_counts)

valid_label_counts = valid_data['potential_labels'].value_counts(normalize=True)
print("\n valid:")
print(valid_label_counts)

test_label_counts = test_data['potential_labels'].value_counts(normalize=True)
print("\n test:")
print(test_label_counts)


target = 'potential_labels'

X_train = train_data.drop(columns=['u_userId','potential_labels'])
y_train = train_data[target]
X_valid = valid_data.drop(columns=['u_userId','potential_labels'])
y_valid = valid_data[target]
X_test = test_data.drop(columns=['u_userId','potential_labels'])
y_test = test_data[target]


train_pool = Pool(data=X_train, label=y_train)
valid_pool = Pool(data=X_valid, label=y_valid)
test_pool = Pool(data=X_test, label=y_test)


model = CatBoostClassifier(iterations=10000,
                           learning_rate=0.1,
                           depth=6,
                           loss_function='Logloss',
                           eval_metric='Accuracy',
                           verbose=10,
                           early_stopping_rounds=10
                           )


model.fit(train_pool, eval_set=valid_pool)
model.save_model('catboost_model2.json')


metrics = model.get_evals_result()
train_accuracy = metrics['learn']['Accuracy']
valid_accuracy = metrics['validation']['Accuracy']

y_pred = model.predict(test_pool)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
print("Confusion Matrix:")
print(cm_df)


plt.plot(train_accuracy, label='Train')
plt.plot(valid_accuracy, label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.show()


feature_importance = model.get_feature_importance(train_pool)
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
top_20_features = importance_df.head(20)
print("Top 20 Feature Importances:")
print(top_20_features)