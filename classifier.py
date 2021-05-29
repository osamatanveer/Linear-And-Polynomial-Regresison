import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_X = pd.read_csv('question-2-features.csv') # 21613 rows x 4 columns sqft_living, sqft_lot, floors, sqft_above
df_X_sqftliving = df_X.drop(["sqft_lot", "floors", "sqft_above"], axis=1)
df_Y = pd.read_csv('question-2-labels.csv') # 21613 rows x 1 columns price
X = df_X.to_numpy()
y = df_Y = df_Y.to_numpy()

print(np.linalg.matrix_rank(np.matmul(X.T, X)))

X = df_X_sqftliving.to_numpy()
X = np.c_[np.ones((X.shape[0], 1)), X]
XT_X = np.matmul(X.T, X)
XT_Y = np.matmul(X.T, y)
XT_X = np.linalg.inv(XT_X)
w = np.matmul(XT_X, XT_Y)
print("w0 " + str(w[0][0]))
print("w1 " + str(w[1][0]))

plt.title("sqft_living vs price")
plt.xlabel("sqft_living")
plt.ylabel("price")
plt.plot(df_X_sqftliving.to_numpy(), y, 'r+')
plt.plot(X, w[0][0] + w[1][0] * X)
plt.savefig("img.png")

mse = np.matmul(X, w)
mse = y - mse
mse = np.linalg.norm(mse)
mse = mse**2 / X.shape[0]
print(mse)

X = np.c_[np.ones((X.shape[0], 1)), df_X_sqftliving.to_numpy(), np.square(df_X_sqftliving.to_numpy())]

print(X.shape)
print(y.shape)

XT_X = np.matmul(X.T, X)
XT_Y = np.matmul(X.T, y)
XT_X = np.linalg.inv(XT_X)
w = np.matmul(XT_X, XT_Y)
print("w0 " + str(w[0][0]))
print("w1 " + str(w[1][0]))
print("w2 " + str(w[2][0]))

mse = np.matmul(X, w)
mse = y - mse
mse = np.linalg.norm(mse)
mse = mse**2 / X.shape[0]
print(mse)

X_graph

plt.title("sqft_living vs price")
plt.xlabel("sqft_living")
plt.ylabel("price")
X_graph = np.unique(df_X_sqftliving.to_numpy().flatten())
plt.plot(df_X_sqftliving.to_numpy().flatten(), y, 'r+')
y_pred = []
for i in range(X_graph.shape[0]):
  y_pred.append(w[0][0] + (w[1][0] * X_graph[i]) + (w[2][0] * np.square(X_graph[i]))) 

plt.plot(X_graph, y_pred)
plt.savefig("2.png")
plt.show()