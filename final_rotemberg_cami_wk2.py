import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Import tqdm for progress bar

class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()
        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))
        self.w = tf.Variable(rng.normal(shape=[num_inputs, num_outputs], stddev=stddev), trainable=True)
        self.bias = bias
        if self.bias:
            self.b = tf.Variable(tf.zeros(shape=[1, num_outputs]), trainable=True)

    def __call__(self, x):
        z = x @ self.w
        if self.bias:
            z += self.b
        return z

class MLP(tf.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, hidden_activation=tf.nn.relu, output_activation=tf.sigmoid):
        self.layers = []
        input_size = num_inputs
        for _ in range(num_hidden_layers):
            self.layers.append(Linear(input_size, hidden_layer_width))
            input_size = hidden_layer_width
        self.output_layer = Linear(hidden_layer_width, 1)
        self.hidden_activation = hidden_activation  # ReLU for hidden layers
        self.output_activation = output_activation  # Sigmoid for binary output
        
    def __call__(self, x):
        # Forward pass through hidden layers
        for layer in self.layers:
            x = self.hidden_activation(layer(x))  # ReLU in hidden layers
        
        # Forward pass through output layer
        output = self.output_activation(self.output_layer(x))
        
       
        if output.shape[-1] == 1:
            output = tf.squeeze(output, axis=-1)  # Make sure binary classification outputs are 1D
        
        return output

def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    return -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)

# made a classifier with numpy so had to make it take in tensors 
class MLPClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, num_iters, step_size, decay_rate, batch_size, refresh_rate):
        self.model = model
        self.num_iters = num_iters
        self.step_size = step_size
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.is_fitted_ = False
        self.classes_ = np.array([0, 1])  # Binary classification
        self.refresh_rate = refresh_rate

    def fit(self, X, y):
        # Convert inputs to TensorFlow tensors
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        rng = tf.random.Generator.from_seed(42)
        num_samples = X.shape[0]
        step_size = self.step_size
        bar = tqdm(range(self.num_iters))
        # Training loop
        for i in bar:
            batch_indices = rng.uniform(shape=[self.batch_size], maxval=num_samples, dtype=tf.int32)
            x_batch = tf.gather(X, batch_indices)
            y_batch = tf.gather(y, batch_indices)
            
            with tf.GradientTape() as tape:
                y_pred = self.model(x_batch)
                loss = binary_cross_entropy_loss(y_batch, y_pred)

            grads = tape.gradient(loss, self.model.trainable_variables)
            grad_update(step_size, self.model.trainable_variables, grads)

            step_size *= self.decay_rate
            if i % self.refresh_rate == (self.refresh_rate - 1):
                bar.set_description(
                    f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
                )
                bar.refresh()

        # was having issues with the tensor flow variables fitting
        self.is_fitted_ = True
        
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        prob = self.model(X)
        return (prob > 0.5).numpy().astype(int)

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        prob = self.model(X).numpy()
        return np.hstack([1 - prob, prob])



# Generate spiral data used https://www.kaggle.com/code/kwispy/spiral-mlp/notebook
def spiral_data(num_points_per_class, noise):
    t = np.linspace(0, 4 * np.pi, num_points_per_class)
    x_spiral_0 = np.vstack([t * np.cos(t), t * np.sin(t)]).T + noise * np.random.randn(num_points_per_class, 2)
    x_spiral_1 = np.vstack([t * np.cos(t + np.pi), t * np.sin(t + np.pi)]).T + noise * np.random.randn(num_points_per_class, 2)
    X = np.vstack([x_spiral_0, x_spiral_1])
    y = np.hstack([np.zeros(num_points_per_class), np.ones(num_points_per_class)])
    return X, y

# Main code execution
if __name__ == "__main__":
    # Generate data
    X, y = spiral_data(num_points_per_class=200, noise=0.1)
    
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    
    model = MLP(num_inputs=2, num_outputs=1, num_hidden_layers=80000, hidden_layer_width=128, hidden_activation=tf.nn.relu, output_activation=tf.sigmoid)

    # Wrap the model with scikit-learn compatible class
    mlp_classifier = MLPClassifierWrapper(model=model, num_iters=3000, step_size=0.01, decay_rate=0.9995, batch_size=128, refresh_rate=10)

   
    mlp_classifier.fit(X, y)

    
    y_pred = mlp_classifier.predict(X)

   
    fig, ax = plt.subplots(figsize=(8, 6))
    
    
    display = DecisionBoundaryDisplay.from_estimator(mlp_classifier, X, response_method="predict", cmap=plt.cm.coolwarm, ax=ax)
    

    
    ax.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], color='red', label='Class 0 (Predicted)')
    ax.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], color='blue', label='Class 1 (Predicted)')

    plt.title("MLP Binary Classification: Spiral Decision Boundary and Predicted Points")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()
