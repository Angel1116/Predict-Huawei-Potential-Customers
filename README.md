# Predict Huawei Potential Customers
In this project, I developed XGBoost and CatBoost combined with bag-of-values encoding to predict Huawei potential customers, achieving 96% accuracy.

## Key Breakthrough
In fact, some data analysts have also attempted to make predictions with this dataset, but achieving an accuracy rate above 90% has proven challenging. After analyzing the data, I found the main issue to be that some variables were non-numeric, making them harder to process. To address this, I made the following adjustments:  

- **Bag-of-values encoder**: I use bag-of-values encoder for encoding to transform non-numeric variables into numerical representations.  
- **Sparse matrix**: I use sparse matrix for recording to reduce memory usage.  

With these methods, I successfully increased the accuracy from 0.883 to 0.956, as shown in the figure below.  

<img src="https://github.com/user-attachments/assets/7089c842-c1c6-46fc-b1e9-c94ea00e722b" width="550"/>

▲Comparison between use and non-use of the bag-of-values encoding



