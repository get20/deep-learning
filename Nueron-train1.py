import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

print(df)

# Attribute Information:
#   0. sepal length in cm
#   1. sepal width in cm
#   2. petal length in cm
#   3. petal width in cm
#   4. class:   
#      -- Iris Setosa
#      -- Iris Versicolour
#      -- Iris Virginica
        
# Prepare dataset for our neural network:
# Class (Setosa or Versicolor)
y = df.iloc[0:100, 4].values
# If it is not Setosa then it is Versicolor
y = np.where(y == 'Iris-setosa', -1, 1)

# Features (sepal length and petal length)
x = df.iloc[0:100, [0,2]].values