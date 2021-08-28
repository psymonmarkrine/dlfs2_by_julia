include("../common/layers.jl") # MatMul, SoftmaxWithLoss
include("../common/python_likes.jl")

mutable struct SimpleSkipGram
    in_layer
    out_layer
    loss_layer1
    loss_layer2
    layers
    params
    grads
    word_vecs
end

function SimpleSkipGram(vocab_size::Integer, hidden_size::Integer)
    V, H = vocab_size, hidden_size

    # 重みの初期化
    W_in = 0.01f0 * randn(Float32, (V, H))
    W_out = 0.01f0 * randn(Float32, (H, V))

    # レイヤの生成
    in_layer  = MatMul(W_in)
    out_layer = MatMul(W_out)
    loss_layer1 = SoftmaxWithLoss()
    loss_layer2 = SoftmaxWithLoss()

    # すべての重みと勾配をリストにまとめる
    layers = [in_layer, out_layer]
    params = typeof(in_layer0.params)([])
    grads  = typeof(in_layer0.grads)([])
    for layer = layers
        append!(params, layer.params)
        append!(grads,  layer.grads)
    end
    
    # メンバ変数に単語の分散表現を設定
    word_vecs = W_in
    return SimpleSkipGram(in_layer, out_layer, loss_layer1, loss_layer2, layers, params, grads, word_vecs)
end

function forward(self::SimpleSkipGram, contexts, target)
    h = forward(self.in_layer, target)
    s = forward(self.out_layer, h)
    l1 = forward(self.loss_layer1, s, getpart(contexts, [:, 1]))
    l2 = forward(self.loss_layer2, s, getpart(contexts, [:, 2]))
    loss = l1 + l2
    return loss
end

function backward(self::SimpleSkipGram, dout=1)
    dl1 = backward(self.loss_layer1, dout)
    dl2 = backward(self.loss_layer2, dout)
    ds = dl1 + dl2
    dh = backward(self.out_layer, ds)
    backward(self.in_layer, dh)
    return
end
