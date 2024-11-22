#!/usr/bin/env python
# coding: utf-8

# In[ ]:


n_bootstraps = 50  # Number of bootstrap iterations
bootstrap_sample_size = int(0.5 * len(X_train_dense))  # Use 50% of the training data per iteration
bootstrap_scores = []

for _ in range(n_bootstraps):
    indices = np.random.choice(len(X_train_dense), size=bootstrap_sample_size, replace=True)
    X_bootstrap, y_bootstrap = X_train_dense[indices], y_train.iloc[indices]
    
    # Train and predict
    lr.fit(X_bootstrap, y_bootstrap)
    y_pred_bootstrap = lr.predict(X_test_dense)
    
    rss = np.sum((y_test - y_pred_bootstrap)**2)
    tss = np.sum((y_test - np.mean(y_test))**2)
    r2_bootstrap = 1 - (rss / tss)
    bootstrap_scores.append(r2_bootstrap)

# Calculate the mean and standard deviation of R²
bootstrap_mean_r2 = np.mean(bootstrap_scores)
bootstrap_std_r2 = np.std(bootstrap_scores)
print(f"Bootstrapping Mean R² Score: {bootstrap_mean_r2:.4f}")

