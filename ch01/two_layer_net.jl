include("../common/layers.jl") # Affine, Sigmoid, SoftmaxWithLoss


mutable struct TwoLayerNet
    params
    layers
    grads
    loss_layer
end

function TwoLayerNet(input_size, hidden_size, output_size)
    I, H, O = input_size, hidden_size, output_size

    # 重みとバイアスの初期化
    W1 = 0.01 * randn(I, H)
    b1 = zeros(1, H)
    W2 = 0.01 * randn(H, O)
    b2 = zeros(1, O)

    # レイヤの生成
    layers = [
        Affine(W1, b1),
        Sigmoid(),
        Affine(W2, b2)
    ]
    loss_layer = SoftmaxWithLoss()

    # すべての重みと勾配をリストにまとめる
    params = []
    grads = []
    for layer = layers
        append!(params, layer.params)
        append!(grads,  layer.grads)
    end

    return TwoLayerNet(params, layers, grads, loss_layer)
end

function predict(self::TwoLayerNet, x)
    for layer in self.layers
        x = forward(layer, x)
    end
    return x
end

function forward(self::TwoLayerNet, x, t)
    score = predict(self, x)
    loss = forward(self.loss_layer, score, t)
    return loss
end

function backward(self::TwoLayerNet, dout=1)
    dout = backward(self.loss_layer, dout)
    for layer = reverse(self.layers)
        dout = backward(layer, dout)
    end
    return dout
end
