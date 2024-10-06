import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.special import gamma

data = pd.read_csv('data_problem2.csv', sep=',', header=None)

# print(data.head())
# print(data.info())
print(data.describe())
# print(data.shape)

row1 = data.iloc[0, :]
row2 = data.iloc[1, :]

# plt.figure(figsize=(10, 5))

class0 =  []
class1 =  []

for feautures, labels in zip(row1, row2):
    if labels == 0:
        class0.append(feautures)
    elif labels == 1:
        class1.append(feautures)


# g = pd.concat([row1, row2])

# Histogram for the first row (Row 0)
    # plt.hist(class0, bins=40, alpha = 0.7, label=("Class 0"))
    # plt.hist(class1, bins=40, alpha = 0.7, label=('Class 1'))
plt.legend()
plt.savefig("Histogram")
# plt.show()

ratio = 0.8 
X_train, X_test, y_train, y_test = train_test_split(row1, row2, test_size=0.2,
                                     random_state=42)

class0_train = X_train[y_train == 0]
class1_train = X_train[y_train == 1]

alpha = 2
n_0 = len(class0_train)
n_1 = len(class1_train)

Beta_hat = (1/n_0 * alpha) * np.sum(class0_train)
mu_hat = (1/n_1) * np.sum(class1_train)
sigma_squared_hat = (1/n_1) * np.sum((class1_train - mu_hat)**2)

print(f'{Beta_hat:.3f}')
print(f'{mu_hat:.3f}')
print(f'{sigma_squared_hat:.3f}')


def gamma_dist(x, alpha, beta):
    p_0 =  (1 / (beta ** alpha * gamma(alpha))) * (x ** (alpha - 1)) * np.exp(-x / beta)
    return p_0

def gaussian_dist(x, mu, sigma_squared):
    p_1 =  (1 / np.sqrt(2 * np.pi * sigma_squared)) * np.exp(-0.5 * ((x - mu) ** 2) / sigma_squared)
    return p_1




y_pred = []
for x in X_test:
    # Likelihood (Gamma)
    p_x_given_C0 = gamma_dist(x, alpha, Beta_hat)
    # Likelihood (Gaussian)
    p_x_given_C1 = gaussian_dist(x, mu_hat, sigma_squared_hat)
    # Bayes classifier
    if p_x_given_C0 > p_x_given_C1:
        y_pred.append(0)
    else:
        y_pred.append(1)

y_pred = np.array(y_pred)


accuracy = np.mean(y_pred == y_test)
print(f'The accuracy is {accuracy*100:.2f} percent ')

misclassified = (y_pred != y_test)
classified = (y_pred == y_test)
X_test_array = np.array(X_test)

plt.scatter(X_test_array[misclassified], y_test[misclassified], color='red', label='misclassified')
plt.scatter(X_test_array[classified], y_test[classified], color='green', label='classified')
# plt.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
plt.title('Missclassified points')
plt.legend()
plt.savefig('Missclassified points')
plt.show()

