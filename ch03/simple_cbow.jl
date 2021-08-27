include("../common/layers.jl") # MatMul, SoftmaxWithLoss
include("../common/python_likes.jl")

mutable struct SimpleCBOW
    in_layer0
    in_layer1
    out_layer
    loss_layer
    layers
    params
    grads
    word_vecs
end

function SimpleCBOW(vocab_size::Integer, hidden_size::Integer)
    V, H = vocab_size, hidden_size

    # 重みの初期化
    W_in = 0.01f0 * randn(Float32, (V, H))
    W_out = 0.01f0 * randn(Float32, (H, V))

    # レイヤの生成
    in_layer0 = MatMul(W_in)
    in_layer1 = MatMul(W_in)
    out_layer = MatMul(W_out)
    loss_layer = SoftmaxWithLoss()

    # すべての重みと勾配をリストにまとめる
    layers = [in_layer0, in_layer1, out_layer]
    params = typeof(in_layer0.params)([])
    grads  = typeof(in_layer0.grads)([])
    for layer = layers
        append!(params, layer.params)
        append!(grads,  layer.grads)
    end
    # メンバ変数に単語の分散表現を設定
    word_vecs = W_in
    return SimpleCBOW(in_layer0, in_layer1, out_layer, loss_layer, layers, params, grads, word_vecs)
end

function forward(self::SimpleCBOW, contexts, target)
    h0 = forward(self.in_layer0, getpart(contexts, [:, 1]))
    h1 = forward(self.in_layer1, getpart(contexts, [:, 2]))
    h = (h0 + h1) * 0.5
    score = forward(self.out_layer, h)
    loss  = forward(self.loss_layer, score, target)
    return loss
end

function backward(self::SimpleCBOW, dout=1)
    ds = backward(self.loss_layer, dout)
    da = backward(self.out_layer, ds)
    da *= 0.5
    backward(self.in_layer1, da)
    backward(self.in_layer0, da)
end
    