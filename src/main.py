import pandas as pd
import numpy as np

np.random.seed(42)
data = {
    'Sex': np.random.choice(['male','female'],100),
    'Housing': np.random.choice(['own','rent','free'],100),
    'Saving accounts': np.random.choice(['little','moderate','rich','quite rich'],100),
    'Checking account': np.random.choice(['little','moderate','rich','no'],100),
    'Purpose': np.random.choice(['car','furniture','radio/TV','education','business'],100),
    'Credit amount': np.random.randint(500,5000,100),
    'Duration': np.random.randint(4,60,100),
    'Risk': np.random.choice(['good','bad'],100)
}
df = pd.DataFrame(data)
df.to_csv('data/german_credit_data.csv', index=False)
