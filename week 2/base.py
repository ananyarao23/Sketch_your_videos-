import numpy as np
import random
import logging
import pickle

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


class NeuralNetwork:

    def __init__(self, num_layers, num_neurons_per_layer_list, seed=400):
        logging.info("Creating a NN with {} layers having {} neurons respectively".format(num_layers,
                                                                                          num_neurons_per_layer_list))
        seed_everything(seed)
        self.seed = seed
        self.num_layers = num_layers
        self.num_neurons_per_layer_list = num_neurons_per_layer_list
        # The following self.weights and self.biases list contains layer-by-layer weights and biases as numpy arrays
        # For example, if we have an input layer with 784 neurons, two hidden layer with 8 neurons, and an output layer
        # with 10 neurons:
        # self.biases = [numpy array (8, 1), numpy array (8, 1), numpy array (10, 1)]
        # self.weights = [numpy array (784, 8), numpy array (8, 8), numpy array (8, 10)]
        self.biases = [np.random.randn(i, 1) for i in num_neurons_per_layer_list[1:]]
        self.weights = [np.random.randn(i, j)
                        for i, j in zip(num_neurons_per_layer_list[:-1], num_neurons_per_layer_list[1:])]
        pass

    @staticmethod
    def sigmoid(z):
        """
        :param z: numpy array (number of neurons, 1)
        :return: numpy array (number of neurons, 1)
        """
        sigmoid_z = np.zeros(z.shape)
        # TODO implement sigmoid
        sigmoid_z = 1/(1+np.exp(-z))
        # TODO implement sigmoid
        return sigmoid_z

    @staticmethod
    def cost_derivative(output_activations, y):
        """
        Derivative of cost w.r.t final activations
        :param output_activations: numpy array (number of neurons in output layer, 1)
        :param y: numpy array (number of neurons in output layer, 1)
        :return: numpy array (number of neurons in output layer, 1)
        """
        derivative_cost = np.zeros(output_activations.shape)
        # TODO calculate Derivative of cost w.r.t final activation
        derivative_cost = output_activations - y
        # TODO calculate Derivative of cost w.r.t final activation
        return derivative_cost

    def cost(self, training_data):
        """ Calculate cost (sum of squared differences) over training data for the current weights and biases
        :param training_data:
        list of 50000 samples, each item of the list is a tuple (x, y)
        x - input numpy array (784, 1)  y - labelled output numpy array (10, 1)
        :return: cost - float
        """
        logging.info("Calcuting costs...")
        cost = 0
        # TODO calculate costs
        # input_data = np.array(list(map(lambda x: x[0], training_data)))
        # output_data = np.array(list(map(lambda x: x[1], training_data)))
        # # y_out = np.zeros((self.num_neurons_per_layer_list[-1], 1))
        # X = input_data.T
        # for j in range(self.num_layers-1):
        #     z_j = np.matmul(np.transpose(self.weights[j]),X) + self.biases[j]
        #     a_j = np.zeros((z_j.shape[0],z_j.shape[1]))
        #     for i in range(len(input_data)):
        #         a_j[i] = self.sigmoid(z_j[i]).copy()
        #     X = a_j
        # y_out = X.T
        # diff = y_out - output_data
        # sq_diff = diff**2
        # cost = np.sum(sq_diff)/2
        for data in training_data:
            diff = (self.forward_pass(data[0]) - data[1])
            diff = (diff**2)/2
            cost += np.sum(diff)
        # TODO calculate costs
        logging.info("Calcuting costs complete...")
        return cost

    def sigmoid_derivative(self, z):
        """Derivative of the sigmoid function."""
        derivative_sigmoid = np.zeros(z.shape)
        # TODO calculate derivative of sigmoid function
        derivative_sigmoid = self.sigmoid(z)*(1-self.sigmoid(z))
        # TODO calculate derivative of sigmoid function
        return derivative_sigmoid

    def forward_pass(self, x):
        """
        Perform forward pass and return the output of the NN.
        :param x: numpy array (784, 1)
        :return: numpy array (10, 1)
        """
        nn_output = np.zeros((self.num_neurons_per_layer_list[-1], 1))
        # TODO do a forward pass of the NN and return the final output
        input_data = x
        for j in range(self.num_layers-1):
            z_j = np.matmul(np.transpose(self.weights[j]),input_data) + self.biases[j]
            a_j = self.sigmoid(z_j)
            input_data = a_j
        nn_output = input_data
        # TODO do a forward pass of the NN and return the final output
        return nn_output

    def mini_batch_GD(self, training_data, epochs, mini_batch_size, eta, test_data):
        """ Train the neural network using mini batch gradient descent
        :param training_data: list of tuples, where each tuple is a single labelled data point as follows.
         "(x - numpy array (784, 1), y (10, 1))"
        :param epochs: int
        :param mini_batch_size: int
        :param eta: learning rate float
        :param test_data: test data
        :return: cost_history - list (append cost after every epoch to this list, this will used to reward marks for
        gradient descent update step)
        """
        logging.info("Running mini_batch_GD")
        num_samples = len(training_data)
        costs_history = []
        for j in range(epochs):
            # print(f"-----------------EPOCH {j}------------------")
            random.Random(self.seed).shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, num_samples, mini_batch_size)]
            num_batch = 1
            for mini_batch in mini_batches:
                # print(f"-----------------BATCH PROCESSING {num_batch}-----------------")
                num_batch += 1
                # TODO (1) Compute gradients for every data point in the mini batch using your implemented
                #  "back_propagation" function. Note that the above backward pass is for a single data point.

                dC_db_total = [np.zeros(b.shape) for b in self.biases]
                dC_dw_total = [np.zeros(w.shape) for w in self.weights]
                input__x = [mini_batch[i][0].reshape(-1,) for i in range(len(mini_batch))]
                output_y = [mini_batch[i][1].reshape(-1,) for i in range(len(mini_batch))]
                for i in range(len(mini_batch)):
                    input = input__x[i].reshape(-1,1)
                    output = output_y[i].reshape(-1,1)
                    x,y = self.back_propagation(input,output)
                    for j in range(len(dC_db_total)):
                        dC_db_total[j] += x[j]
                        dC_dw_total[j] += y[j]
                #  We'll need to calculate this for all examples in a mini batch and then sum the gradients over this
                #  batch to do the gradient update step for all weights and biases.
                # TODO (2) Update biases and weights using the computed gradients
                for j in range(len(self.weights)):
                    self.weights[j] -= int((eta/mini_batch_size))*dC_dw_total[j]
                    self.biases[j] -= int(eta/mini_batch_size)*dC_db_total[j]

            logging.info("After Epoch {}".format(j + 1))
            self.test_accuracy(test_data)
            cost_curr = self.cost(training_data)
            costs_history.append(cost_curr)
            logging.info("Cost: {}".format(cost_curr))
        return costs_history

    def back_propagation(self, x, y):
        """Compute gradients
        :param x: Input to the NN - numpy array (784, 1)
        :param y: Labelled output data - numpy array (10, 1)
        :return: tuple (dC_db, dC_dw) - list of numpy arrays, similar to self.biases and self.weights
        """
        # These list contain layer-by-layer gradients as numpy arrays, similar to self.biases and self.weights
        # For example, if we have an input layer with 784 neurons, two hidden layer with 8 neurons, and an output layer
        # with 10 neurons:
        # dC_db = [numpy array (8, 1), numpy array (8, 1), numpy array (10, 1)]
        # dC_dw = [numpy array (784, 8), numpy array (8, 8), numpy array (8, 10)]
        dC_db = [np.zeros(b.shape) for b in self.biases]
        dC_dw = [np.zeros(w.shape) for w in self.weights]

        # TODO (1) forward pass - calculate layer by layer z's and activations which will be used to calculate gradients
        # TODO (2) backward pass - calculate gradients starting with the output layer and then the hidden layers
        # TODO (3) Return the graduents in lists dC_db, dC_dw
        Z = []
        A = []
        z = x
        # print("num_layers",self.num_layers)
        a = x
        A.append(x)
        for i in range(self.num_layers-1):
            # A.append(a)
            z = np.matmul(self.weights[i].T,z) + self.biases[i]
            a = self.sigmoid(z)
            A.append(a)
            Z.append(z)
            z = a

        # -----------------------------------------------------
        dz_dW = np.array([A[-2].reshape(-1) for j in range(dC_dw[-1].shape[1])])
        temp1 = self.cost_derivative(A[-1],y)*self.sigmoid_derivative(Z[-1])
        dC_dw[-1] = (temp1*dz_dW).T
        dC_db[-1] = temp1

        for j in range(self.num_layers-3,-1,-1):
            temp1 = temp1.reshape(1,-1)
            temp = np.matmul(temp1,self.weights[j+1].T).reshape(-1,1)
            temp1 = temp*(self.sigmoid_derivative(Z[j]))
            dz_dW = np.array([A[j].reshape(-1) for i in range(dC_dw[j].shape[1])])
            dC_dw[j] = (temp1*dz_dW).T
            dC_db[j] = temp1

      
        return dC_db, dC_dw

    def test_accuracy(self, test_data):
        acc_val = 0
        for x, y in test_data:
            if y == np.argmax(self.forward_pass(x)):
                acc_val += 1
        logging.info("Test accuracy {}".format(round(acc_val / len(test_data) * 100, 2)))


def load_data(file="data.pkl"):
    def vectorize(j):
        y = np.zeros((10, 1))
        y[j] = 1.0
        return y

    logging.info("loading data...")
    with open(file, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
        training_outputs = [vectorize(y) for y in training_data[1]]
        training_data = list(zip(training_inputs, training_outputs))
        test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
        test_data = list(zip(test_inputs, test_data[1]))
    logging.info("loaded data...")
    return training_data, test_data


def seed_everything(seed=0):
    np.random.seed(seed)


if __name__ == "__main__":
    data_training, data_test = load_data()
    net = NeuralNetwork(num_layers=4, num_neurons_per_layer_list=[784, 8, 8, 10])
    net.mini_batch_GD(data_training, 15, 50, 0.1, data_test)
