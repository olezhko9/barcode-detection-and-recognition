import numpy as np
import pickle5 as pickle


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, data):
        out = self.layers[0].forward((data / 255) - 0.5)

        for layer in self.layers[1:]:
            out = layer.forward(out)

        return out

    def backprop(self, gradient, lr):
        gradient = self.layers[-1].backprop(gradient, lr)

        for layer in self.layers[-2::-1]:
            gradient = layer.backprop(gradient, lr)

        return gradient

    def train(self, train_data, train_labels, lr=.01, epochs=1):
        for epoch in range(epochs):
            print('--- Epoch %d ---' % (epoch + 1))

            permutation = np.random.permutation(len(train_data))
            train_data = train_data[permutation]
            train_labels = train_labels[permutation]

            loss = 0
            accuracy = 0
            for i, (sample, label) in enumerate(zip(train_data, train_labels)):
                if i % 100 == 99:
                    print(
                        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                        (i + 1, loss / 100, accuracy)
                    )
                    loss = 0
                    accuracy = 0

                predicted = self.forward(sample)
                l = -np.log(predicted[label])
                acc = 1 if np.argmax(predicted) == label else 0

                loss += l
                accuracy += acc

                gradient = np.zeros(10)
                gradient[label] = -1 / predicted[label]

                self.backprop(gradient, lr)

    def predict(self, data):
        return self.forward(data)

    def save(self, weights_file):
        obj = {}
        for layer in self.layers:
            weights = layer.get_weights()
            obj[layer.name] = weights

        with open(weights_file, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, weights_file):
        with open(weights_file, 'rb') as handle:
            weights = pickle.load(handle)
            for layer in self.layers:
                layer.load_weights(weights[layer.name])


if __name__ == '__main__':
    import layers
    from time import time

    n_filters = 8
    lr = 0.01
    epochs = 10

    net = Network()
    net.add_layer(layers.Conv(num_filters=n_filters, kernel_size=3))
    net.add_layer(layers.MaxPool(pool_size=2))
    net.add_layer(layers.Softmax(13 * 13 * n_filters, 10))

    train_images = np.load('../train_x.npy')
    train_labels = np.load('../train_y.npy')

    start = time()
    net.train(train_images, train_labels, lr, epochs)
    net.save('./weights-10.pkl')
    print(time() - start)
