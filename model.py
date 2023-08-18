import numpy as np
import time

class Layer:
    def __init__(self) -> None:
        pass

    def initialize_parameter(self, n_prev, n_layer):
        # n_prev: số neuron của layer trước
        # n_layer: số neuron của layer hiện tại
        W = np.random.randn(n_layer, n_prev)*np.sqrt(1/n_layer)
        b = np.zeros((n_layer, 1))
        return W, b
    
    def compute_Z(self, W, b, A_prev):
        Z = np.dot(W, A_prev) + b
        return Z

    def relu(self, z):
        A = np.maximum(0, z)
        return A
    
    def sigmoid(self, z):
        A = 1/(1+np.exp(-z))
        return A
    '''
    def softmax(self, z):
        z_exp = np.exp(z)
        z_sum = np.sum(z_exp,axis = 1,keepdims = True)
        A = z_exp / z_sum
        return A
    '''
    def softmax(self, z):
        z_exp = np.exp(z - np.max(z, axis=0, keepdims=True))  # Tránh trường hợp số lớn gây overflow
        A = z_exp / np.sum(z_exp, axis=0, keepdims=True)
        return A
    
    def compute_cost(self, A, Y):
        m = Y.shape[1]
        cost = -np.sum(np.multiply(Y, np.log(A)), axis=1, keepdims=True)/m
        cost = round(np.mean(cost),3)
        return cost
    
    def compute_grad(self, A_prev, dZ):
        m = A_prev.shape[0]
        dW = np.dot(dZ, A_prev.T)*1/m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        return dW, db
    
    def compute_dA_prev(self, W, dZ):
        dA_prev = np.dot(W.T, dZ)
        return dA_prev
    
    def compute_dZ(self, dA, A):
        dZ = dA*A*(1-A)
        return dZ
    
    def update_parameters(self, dW, db, W, b, learning_rate):
        new_W = W - learning_rate*dW
        new_b = b - learning_rate*db
        return new_W, new_b
    
class My_ANN_model:
    def __init__(self, layer_dims, learning_rate):
        # layer_dims: danh sách chưa số neuron của từng layer
        # learning_rate: độ dài bước học để phục vụ cho gradient_descent
        self.layer_dims = layer_dims
        self.deep_num = len(layer_dims)     # Số layer của mạng
        self.lr = learning_rate
        self.layer = Layer()
        self.parameters = {}

    def fit(self, X, Y):
        # X, Y: tập dữ liệu huấn luyện
        self.X = X.reshape(X.shape[0], -1).T
        self.Y = Y.T
        return

    def initialize_parameters(self):
        A = self.X.shape[0]
        for neuron in range(self.deep_num):
            W, b = self.layer.initialize_parameter(A, self.layer_dims[neuron])
            self.parameters['W'+str(neuron+1)] = W
            self.parameters['b'+str(neuron+1)] = b
            A = self.layer_dims[neuron]
        return self.parameters
    
    def forward_propagation(self):
        if self.parameters is None:
            self.initialize_parameters()
        cache = {}
        A_prev = self.X
        for layer in range(1, self.deep_num+1):
            W, b = self.parameters['W'+str(layer)], self.parameters['b'+str(layer)]
            Z = self.layer.compute_Z(W, b, A_prev)
            if layer==self.deep_num:    # Layer cuối
                A = self.layer.softmax(Z)
            else:
                A = self.layer.sigmoid(Z)
            cache['A'+str(layer-1)] = A_prev
            A_prev = A
        cost = self.layer.compute_cost(A, self.Y)
        return cache, A, cost

    def backward_propagation(self, cache, A):
        dZ = A - self.Y
        for layer in reversed(range(self.deep_num)):
            W, b = self.parameters['W'+str(layer+1)], self.parameters['b'+str(layer+1)]
            dW, db = self.layer.compute_grad(cache['A'+str(layer)], dZ)
            new_W, new_b = self.layer.update_parameters(dW, db, W, b, self.lr)
            dA_prev = self.layer.compute_dA_prev(dZ, new_W)
            self.parameters['W'+str(layer+1)] = new_W
            self.parameters['b'+str(layer+1)] = new_b
            dZ = self.layer.compute_dZ(dA_prev.T, cache['A'+str(layer)])
        return

    def training(self, epochs):
        for epoch in range(epochs):
            start_time = time.time()
            cache, A, cost = self.forward_propagation()
            self.backward_propagation(cache, A)
            end_time = time.time()
            print(f'Epoch {epoch+1} has been trained.-----------------> Time: {round(end_time-start_time,2)}s - Cost: {cost}')
        return

    