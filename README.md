# Project 2
------
# README - Flight Prediction

Roger Kewin Samson - A20563057

Ekta Shukla - A20567127

Rithika Kavitha Suresh - A20564346

Jude Rosun - A20564339

Analysis Overview
This project evaluates different model selection techniques for predicting flight ticket prices. Specifically, we applied cross-validation, bootstrapping, and Akaike Information Criterion (AIC) for model evaluation and selection in a linear regression setting.

Results Summary
1. K-Fold Cross-Validation:
   - Mean (R^2): 0.9115

2. Bootstrapping:
   - Mean (R^2): 0.9113

3. AIC (Akaike Information Criterion):
   - AIC was calculated based on the residual sum of squares (RSS) for model evaluation.


Question 1: Do Cross-Validation and Bootstrapping Agree with Simpler Model Selectors Like AIC in Simple Cases?
Answer -
Yes, in this linear regression case, cross-validation and bootstrapping provide similar 

  estimates, both indicating a high degree of model fit ACCURACY - 9.11. This agreement suggests that all three methods—cross-validation, bootstrapping, and AIC—are consistent in simple regression scenarios.

Observations from Different Models:
We experimented with advanced models to evaluate their performance:
Using raw GPU data, the accuracy was only 60% due to poor data quality.
After data cleaning, the model improved to 76% accuracy.
Implementing Lasso and Ridge regression yielded similar results to linear regression, with accuracy around 76%, showing that regularization wasn’t necessary for this case.
Importing pre-cleaned data with better features and ensuring proper preprocessing improved the model performance further, achieving consistent results.
Focus on K-Fold Cross-Validation and Bootstrapping:
Cross-Validation: Directly evaluates the generalization error by dividing the data into folds, ensuring unbiased performance estimates. It highlighted areas for improvement when the model was overfitting.
Bootstrapping: Approximated the distribution of accuracy 
  scores, showing the robustness of the model and its reliability across multiple resampling iterations.
AIC (Akaike Information Criterion): Provided a simpler metric by penalizing overly complex models, confirming the linear regression model as the most appropriate choice.
Key Takeaways:
Cross-validation and bootstrapping both agreed with AIC, especially in this linear regression case, demonstrating consistency in their evaluation of the model's performance.
This consistency ensures that the project focuses on not only implementing these advanced evaluation techniques but also on achieving reliable results using these methods.
By cleaning the data and testing with advanced methods, we confirmed that cross-validation and bootstrapping align with AIC in simple regression scenarios, validating their reliability as model selection techniques.

Question 2: In What Cases Might These Methods Fail or Give Incorrect/Undesirable Results?
Answer - 
Cross-validation, bootstrapping, and AIC can fail or produce undesirable results in specific cases. Cross-validation may yield overly optimistic results if feature selection or preprocessing is performed before splitting the data, as this leaks information from test folds into training folds. Additionally, small datasets can result in high variance in fold splits, making performance estimates less reliable. Unclean or unsuitable data further exacerbates these issues, especially when proper preprocessing steps or required libraries are skipped. Bootstrapping, on the other hand, can fail when working with very small datasets or those that do not represent the population accurately, as resampling bias can skew results. It is also computationally intensive, especially with large datasets or a high number of iterations, though adjusting parameters can improve runtime at the cost of robustness. Finally, AIC assumes the model is correctly specified and does not account for prediction error as cross-validation does. This limitation may lead AIC to favor overly simple models that fail to capture the data's complexity, particularly when the model is misspecified or the data is intricate.

Question 3: What Could Be Implemented to Mitigate These Cases or Help Users of These Methods?
Answer - 
Cross-validation, bootstrapping, and AIC are powerful tools for model evaluation, but there are ways to improve their robustness and usability. Nested cross-validation could ensure unbiased performance estimates by separating feature selection and evaluation processes, while stratified splits and bootstrapping would help maintain the balance and representativeness of the data, especially in imbalanced scenarios. To handle computational challenges, parallelizing bootstrapping would make it feasible for larger datasets. For AIC, adding complementary metrics like BIC or diagnostics such as residual plots could provide deeper insights into model performance and assumptions. These enhancements would make the methods more reliable and user-friendly in diverse scenarios. 

4. What Parameters Have You Exposed to Users for Model Selection?
Answer - 
Users have control over several parameters to customize how the model is evaluated. For K-Fold Cross-Validation, they can set the number of folds (n_splits) to decide how the data is split for testing and training. In bootstrapping, they can adjust the number of iterations (n_bootstraps) and the size of each sample (bootstrap_sample_size) to balance between accuracy and runtime. Users can also choose which features to preprocess, such as categorical features (using one-hot encoding) and numerical features (scaled to have a mean of zero and standard deviation of one). If they want to test non-linear relationships, they can add polynomial features by specifying the degree (degree). These options let users fine-tune the model based on their dataset and requirements.


Explanation of Results:
The high (R^2) values from both cross-validation and bootstrapping indicate that the model explains over 91% of the variance in flight prices, demonstrating good performance. The near-zero standard deviation in bootstrapping implies stability in predictions across different subsets.

Practical Implications:
- Cross-validation and bootstrapping provide robust insights into model performance and generalization.
- AIC complements these methods by penalizing overfitting, ensuring a balance between complexity and fit.

