#!/usr/bin/env python
# coding: utf-8

# In[ ]:


kf = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = []

for train_index, test_index in kf.split(X_train_dense):
    # Split data manually for each fold
    X_train_kf, X_test_kf = X_train_dense[train_index], X_train_dense[test_index]
    y_train_kf, y_test_kf = y_train.iloc[train_index], y_train.iloc[test_index]
    
    lr.fit(X_train_kf, y_train_kf)
    y_pred_kf = lr.predict(X_test_kf)
    
    rss = np.sum((y_test_kf - y_pred_kf)**2)
    tss = np.sum((y_test_kf - np.mean(y_test_kf))**2)
    r2_kf = 1 - (rss / tss)
    cv_scores.append(r2_kf)

# Calculate the mean R² score across folds
cv_mean_r2 = np.mean(cv_scores)
print(f"K-Fold Cross-Validation Mean R² Score: {cv_mean_r2:.4f}")

