import numpy as np


class Conv:
    def __init__(self, num_filters, kernel_size=3, name='conv'):
        self.num_filters = num_filters
        self.size = kernel_size
        self.name = name

        self.filters = np.random.randn(num_filters, self.size, self.size) / 9

    def iterate_regions(self, image):
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + self.size), j:(j + self.size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # d_L_d_input = np.zeros(self.last_input.shape)
        # for f in range(self.num_filters):
        #     kernel_rot180 = np.rot90(self.filters[f], 2)
        #     for img_region, i, j in self.iterate_regions(d_L_d_out[f]):
        #         try:
        #             d_L_d_input[i, j, f] += np.sum(kernel_rot180 * img_region)
        #         except IndexError:
        #             d_L_d_input[i, j] += np.sum(kernel_rot180 * img_region)

        self.filters -= learn_rate * d_L_d_filters

        # return d_L_d_input
        return None


class MaxPool:
    def __init__(self, pool_size=2, name='pool'):
        self.size = pool_size
        self.name = name

    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * self.size):(i * self.size + self.size),
                            (j * self.size):(j * self.size + self.size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backprop(self, d_L_d_out, lr):
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input


class Softmax:
    def __init__(self, input_len, output_len, name='softmax'):
        self.weights = np.random.randn(input_len, output_len) / input_len
        self.biases = np.zeros(output_len)
        self.name = name

    def forward(self, input):
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)

            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_L_d_t = gradient * d_out_d_t

            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)
