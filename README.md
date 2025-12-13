# Credit Risk Prediction + Smart Recovery System (ML + AI Logic)

**Level:** Intermediate  
**Technologies:** Python, Pandas, Scikit-learn, Random Forest

## Project Goal
Build a machine learning system that:
- Predicts the **Probability of Default (PD)** for a loan applicant.
- Provides **smart recovery recommendations** to collection managers based on risk level and borrower profile.

This helps financial institutions reduce losses by prioritizing high-risk cases and choosing the optimal contact channel.

## Dataset
We use the classic **German Credit Data** dataset (1000 samples, 20 features).

Download the preprocessed CSV version from Kaggle:  
https://www.kaggle.com/datasets/uciml/german-credit

Place it in the `data/` folder as `german_credit_data.csv`.

Target variable: `Risk` ('good' = 0, 'bad' = 1 → default).

## Project Structure
