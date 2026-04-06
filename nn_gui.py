import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------
# 1. Generate synthetic sales data
# ---------------------------
np.random.seed(42)
n_samples = 2000


day = np.random.randint(0, 7, n_samples)
day_sin = np.sin(2 * np.pi * day / 7)
day_cos = np.cos(2 * np.pi * day / 7)


food = np.random.uniform(50, 500, n_samples)
beer = np.random.uniform(20, 300, n_samples)
wine = np.random.uniform(10, 200, n_samples)
liquor = np.random.uniform(5, 150, n_samples)


weekend_boost = (day >= 5) * 0.2
interaction = 0.05 * (food * beer) / 1000
noise = np.random.normal(0, 10, n_samples)
total_sales = (food + beer + wine + liquor) * (1 + weekend_boost) + interaction + noise


X = np.column_stack([day_sin, day_cos, food, beer, wine, liquor])
y = total_sales.reshape(-1, 1)


print("Input shape:", X.shape)
print("Sales range: {:.1f} to {:.1f}".format(y.min(), y.max()))


# ---------------------------
# 2. Train / test split & scaling
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)


# ---------------------------
# 3. Multi‑layer neural network (NumPy)
# ---------------------------
class MultiLayerNN:
   def __init__(self, layer_dims, learning_rate=0.01):
       self.lr = learning_rate
       self.L = len(layer_dims) - 1
       self.params = {}
       for l in range(1, self.L + 1):
           self.params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
           self.params[f'b{l}'] = np.zeros((layer_dims[l], 1))


   def relu(self, Z):
       return np.maximum(0, Z)


   def relu_derivative(self, Z):
       return (Z > 0).astype(float)


   def forward(self, X):
       caches = {}
       A = X
       for l in range(1, self.L + 1):
           W = self.params[f'W{l}']
           b = self.params[f'b{l}']
           Z = np.dot(W, A) + b
           if l == self.L:
               A = Z
           else:
               A = self.relu(Z)
           caches[f'Z{l}'] = Z
           caches[f'A{l}'] = A
       return A, caches


   def backward(self, X, y, caches):
       m = X.shape[1]
       grads = {}
       A_last = caches[f'A{self.L}']
       dA_last = 2 * (A_last - y)
       for l in reversed(range(1, self.L + 1)):
           Z = caches[f'Z{l}']
           if l == self.L:
               dZ = dA_last
           else:
               dA = grads[f'dA{l+1}']
               dZ = dA * self.relu_derivative(Z)
           A_prev = X if l == 1 else caches[f'A{l-1}']
           m = A_prev.shape[1]
           grads[f'dW{l}'] = (1/m) * np.dot(dZ, A_prev.T)
           grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
           if l > 1:
               grads[f'dA{l}'] = np.dot(self.params[f'W{l}'].T, dZ)
       for l in range(1, self.L + 1):
           self.params[f'W{l}'] -= self.lr * grads[f'dW{l}']
           self.params[f'b{l}'] -= self.lr * grads[f'db{l}']


   def train_step(self, X, y):
       y_pred, caches = self.forward(X)
       loss = np.mean((y_pred - y) ** 2)
       self.backward(X, y, caches)
       return loss


   def predict(self, X):
       y_pred, _ = self.forward(X)
       return y_pred


# ---------------------------
# 4. Live GUI training
# ---------------------------
def run_gui():
   nn = MultiLayerNN(layer_dims=[6, 64, 32, 16, 1], learning_rate=0.01)


   epochs = 100
   batch_size = 64
   loss_history = []


   X_train_T = X_train.T
   y_train_T = y_train.T
   X_test_T = X_test.T
   y_test_T = y_test.T


   plt.ion()
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   fig.suptitle("Multi‑Layer Neural Network – Sales Prediction (live)", fontsize=14)


   loss_line, = ax1.plot([], [], 'b-')
   ax1.set_xlabel("Batch")
   ax1.set_ylabel("Loss (MSE)")
   ax1.set_title("Training Loss")
   ax1.grid(True)


   scat = ax2.scatter([], [], alpha=0.5)
   ax2.set_xlabel("True Sales (standardised)")
   ax2.set_ylabel("Predicted Sales (standardised)")
   ax2.set_title("Test Set: Prediction vs True")
   ax2.plot([-3, 3], [-3, 3], 'r--')
   ax2.set_xlim(-2.5, 2.5)
   ax2.set_ylim(-2.5, 2.5)


   batch_counter = 0
   for epoch in range(epochs):
       indices = np.random.permutation(X_train_T.shape[1])
       X_shuffled = X_train_T[:, indices]
       y_shuffled = y_train_T[:, indices]
       for i in range(0, X_shuffled.shape[1], batch_size):
           X_batch = X_shuffled[:, i:i+batch_size]
           y_batch = y_shuffled[:, i:i+batch_size]
           loss = nn.train_step(X_batch, y_batch)
           loss_history.append(loss)
           batch_counter += 1
           if batch_counter % 20 == 0:
               loss_line.set_data(range(len(loss_history)), loss_history)
               ax1.relim()
               ax1.autoscale_view()
       y_pred_test = nn.predict(X_test_T)
       scat.set_offsets(np.column_stack([y_test_T.flatten(), y_pred_test.flatten()]))
       test_mse = np.mean((y_pred_test - y_test_T) ** 2)
       ax2.set_title(f"Epoch {epoch+1}/{epochs} – Test MSE: {test_mse:.4f}")
       fig.canvas.draw()
       fig.canvas.flush_events()
       plt.pause(0.01)
       if (epoch+1) % 10 == 0:
           print(f"Epoch {epoch+1}/{epochs}, Last loss: {loss:.6f}")


   plt.ioff()
   plt.show(block=False)  # keep window open


   # ---------------------------
   # 5. Interactive prediction & data addition (CORRECTED)
   # ---------------------------
   print("\n" + "="*50)
   print("END-OF-DAY PREDICTION")
   print("Enter your sales numbers to predict total sales.")
   print("Type 'quit' to exit.\n")


   # We'll keep a local copy of the training data to allow retraining
   X_train_dynamic = X_train.copy()
   y_train_dynamic = y_train.copy()
   scaler_X_dynamic = StandardScaler()
   scaler_y_dynamic = StandardScaler()
   # Re‑fit scalers on current dynamic data
   X_train_dynamic_scaled = scaler_X_dynamic.fit_transform(X_train_dynamic)
   y_train_dynamic_scaled = scaler_y_dynamic.fit_transform(y_train_dynamic)
   # Re‑initialise network and retrain on current data (function to retrain)
   def retrain_model():
       nonlocal nn, X_train_T, y_train_T, X_test_T, y_test_T, scaler_X_dynamic, scaler_y_dynamic
       nn = MultiLayerNN(layer_dims=[6, 64, 32, 16, 1], learning_rate=0.01)
       X_train_T_dyn = X_train_dynamic_scaled.T
       y_train_T_dyn = y_train_dynamic_scaled.T
       # Quick retrain (fewer epochs for speed)
       for epoch in range(50):
           indices = np.random.permutation(X_train_T_dyn.shape[1])
           X_shuffled = X_train_T_dyn[:, indices]
           y_shuffled = y_train_T_dyn[:, indices]
           for i in range(0, X_shuffled.shape[1], 64):
               X_batch = X_shuffled[:, i:i+64]
               y_batch = y_shuffled[:, i:i+64]
               nn.train_step(X_batch, y_batch)
       # Update test data (same test set, but need to transform with new scaler)
       X_test_scaled_dyn = scaler_X_dynamic.transform(X_test)
       y_test_scaled_dyn = scaler_y_dynamic.transform(y_test)
       X_test_T = X_test_scaled_dyn.T
       y_test_T = y_test_scaled_dyn.T
       # Update scatter plot
       y_pred_test = nn.predict(X_test_T)
       scat.set_offsets(np.column_stack([y_test_T.flatten(), y_pred_test.flatten()]))
       fig.canvas.draw()
       print("Model retrained with new data point.")


   while True:
       try:
           day_input = input("Day of week (0=Monday, 1=Tuesday, ..., 6=Sunday): ")
           if day_input.lower() == 'quit':
               break
           day_val = int(day_input)
           if day_val < 0 or day_val > 6:
               print("Day must be 0-6. Try again.")
               continue


           food_val = float(input("Food sales: "))
           beer_val = float(input("Beer sales: "))
           wine_val = float(input("Wine sales: "))
           liquor_val = float(input("Liquor sales: "))


           # Encode day
           day_sin_val = np.sin(2 * np.pi * day_val / 7)
           day_cos_val = np.cos(2 * np.pi * day_val / 7)


           new_data = np.array([[day_sin_val, day_cos_val, food_val, beer_val, wine_val, liquor_val]])
           new_data_scaled = scaler_X_dynamic.transform(new_data)
           new_data_T = new_data_scaled.T


           pred_scaled = nn.predict(new_data_T)
           pred_original = scaler_y_dynamic.inverse_transform(pred_scaled.T).flatten()[0]


           print(f"\n>>> Predicted total sales: ${pred_original:.2f}\n")


           answer = input("Add this data point to training set? (y/n): ").lower()
           if answer == 'y':
               true_total = float(input("What was the actual total sales for this day? "))
               # Add new data point
               X_train_dynamic = np.vstack([X_train_dynamic, new_data])
               y_train_dynamic = np.vstack([y_train_dynamic, [[true_total]]])
               # Re‑scale and retrain
               X_train_dynamic_scaled = scaler_X_dynamic.fit_transform(X_train_dynamic)
               y_train_dynamic_scaled = scaler_y_dynamic.fit_transform(y_train_dynamic)
               retrain_model()
               print("Data point added and model updated.\n")
           else:
               print("Prediction done. Data not added.\n")


       except ValueError:
           print("Invalid input. Please enter numbers only.")


   print("\nExiting. Close the plot window to finish.")


if __name__ == "__main__":
   run_gui()

