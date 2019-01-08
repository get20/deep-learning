import matplotlib.pyplot as plt
%matplotlib inline

eta = 0.1

# make 10 iterations
epochs = 10

[weights, errors] = train(x, y, eta, epochs)
print('Weights: %s' % weights)
plt.scatter(df.iloc[0:100, 0],df.iloc[0:100, 2], c=y)
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

plt.plot(range(1, len(errors)+1), errors, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()