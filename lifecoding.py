# 1. Read Dataset
# 2. Extract valuable features
# 3. Data preprocessing
#   3.1 Remove strings from data
#   3.2 Does Spain is more important than France?(Achtung! Dummy variables trap!)
#   3.3 Informations are equal to each other
# 4. Divide dataset to train and test set
# 5. Create Neural Network
# 6. Compile and Fit
# 7. Let's predict
# 8. Congrats!

import pandas as pd
dataset = pd.read_csv("Churn_Modelling.csv")
print(dataset)