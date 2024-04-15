import random
from micrograd import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class RNN:
    def __init__(self, nin, nhidden, nout):
        self.hidden = Layer(nin + nhidden, nhidden)
        self.out = Layer(nhidden, nout)
        self.nhidden = nhidden

    def __call__(self, x, h=None, return_last=False):

        if h is None:
            h = [Value(0.0) for _ in range(self.nhidden)]

        ys = []
        hs = []
        for xi in x:
            x_model = xi + h
            h = self.hidden(x_model)
            hs.append(h)
            ys.append(self.out(h))

        if return_last:
            return ys[0], hs[0]
        else:
            return ys, hs

    def parameters(self):
        return self.hidden.parameters() + self.out.parameters()


class Conv1D:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernels = [Value(random.uniform(-1, 1))
                        for _ in range(kernel_size)]

    def __call__(self, x):
        assert len(
            x) >= self.kernel_size, "Length of the Input array should be at least the kernel size for CNN"

        i = 0
        res = []
        while i <= len(x) - self.kernel_size:
            res.append(sum(ker * xi for ker,
                           xi in zip(self.kernels, x[i:i+self.kernel_size])))
            i += self.stride
        return res

    def parameters(self):
        return self.kernels


if __name__ == '__main__':

    my_rnn = RNN(1, 5, 1)

    test = [[1], [2], [2.5], [1.8], [-0.5], [0.4]]

    print(my_rnn(test)[0])

    my_conv = Conv1D(4, 1)

    test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(my_conv(test))
