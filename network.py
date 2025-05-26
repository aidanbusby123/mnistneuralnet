from keras.datasets import mnist
import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
       # print(self.weights)

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a)+b)
        #print(a)
        return a
    
    def SGD(self, epochs, mini_batch_size, learning_rate, training_data, test_data):
        random.shuffle(training_data)
        n = len(training_data)

        for j in range (0, epochs):
            print(f"Epoch {j}")
            print(f"{self.evaluate(test_data)}/{len(test_data)}")
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update(mini_batch, learning_rate)

    def update(self, mini_batch, learning_rate):
        nabla_w = [np.zeros(np.shape(w)) for w in self.weights]
        nabla_b = [np.zeros(np.shape(b)) for b in self.biases]

        for x, y in mini_batch:
           # print(x, y)
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
           # print("delta_nabla_w: ")
          #  for dbw in delta_nabla_w : print(np.shape(dbw))
          #  print("\n")
            #print(np.shape(delta_nabla_w[0]))
            #print(np.shape(delta_nabla_b[0]))
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        #print(nabla_w)
        nabla_w = [nw/len(mini_batch) for nw in nabla_w]
        nabla_b = [nb/len(mini_batch) for nb in nabla_b]
        

        self.weights = [( w - nw * learning_rate) for w, nw in zip(self.weights, nabla_w) ]
        self.biases = [( b - nb * learning_rate) for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        #feedforward
        a = x
        acts = [x]
        zs = []
        layers = 0
        delta = []
        for w, b in zip(self.weights, self.biases):
          #  print(np.shape(w))
          #  print(np.shape(np.dot(w,a)))
          #  print(np.shape(a))
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            acts.append(a)
            layers+=1
        
    
        delta = (a - y) * sigmoid_prime(z)
        

        nabla_b = [np.zeros(np.shape(b)) for b in self.biases]
        nabla_w = [np.zeros(np.shape(w)) for w in self.weights]
        #print("Layers")
        #print(layers)
        nabla_b[layers-1] = delta
        nabla_w[layers-1] = np.dot(delta, acts[layers-1].transpose())
       # print(np.shape(nabla_w[layers-1]))

        for l in range (layers, 3, -1):
            delta = np.dot(self.weights[l-1].transpose(), delta) * sigmoid_prime(z[l-2])
            #print(delta)
            nabla_b[l-2] = delta
            nabla_w[l-2] = np.dot(delta, acts[l-2].transpose())

        return (nabla_w, nabla_b)
    

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            np.savez(file, np.array(self.weights, dtype=object),np.array(self.biases, dtype=object))
        print("Saved network")
        print(f"Loaded weights: {[w.shape for w in self.weights]}")
        print(f"Loaded biases: {[b.shape for b in self.biases]}")

    def load(self, filename):
        with np.load(filename, allow_pickle=True) as data:
            self.weights = data['arr_0']
            self.biases = data['arr_1']
        
        print(f"Loaded weights: {[np.shape(w) for w in self.weights]}")
        print(f"Loaded biases: {[np.shape(b) for b in self.biases]}")
        if len(self.weights) != len(self.sizes) - 1:
            raise ValueError("Loaded weights do not match the network architecture.")




def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0-sigmoid(z))


def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return np.where(z > 0, 1, 0)

def vec_val(z):
    e = np.zeros((10, 1))
    e[z] = 1.0
    return e



