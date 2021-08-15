mutable struct Sigmoid
    params
end
Sigmoid() = Sigmoid([])

function forward(self::Sigmoid, x)
    return 1 ./ (1 .+ exp.(-x))
end


mutable struct Affine
    params
end

Affine(W, b) = Affine([W, b])

function forward(self::Affine, x)
    W, b = self.params
    out = x * W .+ b
    return out
end


mutable struct TwoLayerNet
    params
    layers
end

function TwoLayerNet(input_size, hidden_size, output_size)
    I, H, O = input_size, hidden_size, output_size

    # 重みとバイアスの初期化
    W1 = randn(I, H)
    b1 = randn(1, H)
    W2 = randn(H, O)
    b2 = randn(1, O)

    # レイヤの生成
    layers = [
        Affine(W1, b1),
        Sigmoid(),
        Affine(W2, b2)
    ]

    # すべての重みをリストにまとめる
    params = []
    for layer = layers
        append!(params, layer.params)
    end

    return TwoLayerNet(params, layers)
end

function predict(self::TwoLayerNet, x)
    for layer = self.layers
        x = forward(layer, x)
    end
    return x
end


x = randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = predict(model, x)
println(s)
