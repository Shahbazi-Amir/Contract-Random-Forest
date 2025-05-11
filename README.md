# Final Report: Classification Using Random Forest

## **1. Introduction**
In this project, the **Random Forest** algorithm was used for data classification. The main objective was to predict the classes (`no` and `yes`) based on the available features.

## **2. Steps Performed**
1. **Data Preprocessing:**
   - Categorical data were converted to numerical using **One-Hot Encoding**.
   - Imbalanced data were balanced using **SMOTE**.
   - Data were split into **training** (80%) and **test** (20%) sets.

2. **Model Training:**
   - A **Random Forest** model was trained.
   - Model performance was evaluated using metrics such as **Precision**, **Recall**, **F1-score**, and **Accuracy**.

3. **Feature Importance Analysis:**
   - Feature importance was calculated using `model.feature_importances_`.
   - Key features identified include `duration`, `euribor3m`, and `nr.employed`.

4. **Evaluation Results:**
   - **Precision**: Approximately 95% for both classes (`no` and `yes`).
   - **Recall**: Approximately 95% for both classes.
   - **F1-score**: Approximately 95% for both classes.
   - **Accuracy**: 95%.

## **3. Conclusion**
- The **Random Forest** model with default parameters performs well.
- Hyperparameter tuning (using Grid Search) was not performed due to technical issues and high computational cost.
- The results indicate that the model has successfully learned the patterns in the data and is suitable for predicting new data.

## **4. Limitations**
- Due to errors related to `pos_label=1`, hyperparameter tuning was not performed.
- If more resources are available, **Grid Search** or **Randomized Search** can be used in the future.

## **5. Final Outcome**
The current model achieves 95% accuracy and is suitable for predicting new data.