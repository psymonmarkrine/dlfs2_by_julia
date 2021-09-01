include("../common/layers.jl")
include("negative_sampling_layer.jl") # NegativeSamplingLoss


mutable struct SkipGram
    in_layer
    loss_layers
    params
    grads
    word_vecs
end

function SkipGram(vocab_size, hidden_size, window_size, corpus)
    V, H = vocab_size, hidden_size
    
    # 重みの初期化
    W_in = 0.01 * randn(Float32, V, H)
    W_out = 0.01 * randn(Float32, V, H)

    # レイヤの生成
    in_layer = Embedding(W_in)
    loss_layers = [NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5) for _=1:2window_size]

    # すべての重みと勾配をリストにまとめる
    layers = [in_layer, loss_layers...]
    params = typeof(in_layer.params)([])
    grads  = typeof(in_layer.grads)([])
    for layer = layers
        append!(params, layer.params)
        append!(grads,  layer.grads)
    end

    # メンバ変数に単語の分散表現を設定
    word_vecs = W_in
    return SkipGram(in_layer, loss_layers, params, grads, word_vecs)
end

function forward(self::SkipGram, contexts, target)
    h = forward(self.in_layer, target)

    loss = sum([forward(layer, h, selectdim(contexts, 2, i)) for (i, layer)=enumerate(self.loss_layers)])
    return loss
end

function backward(self::SkipGram, dout=1)
    dh = sum([backward(layer, dout) for (i, layer)=enumerate(self.loss_layers)])
    backward(self.in_layer, dh)
end
