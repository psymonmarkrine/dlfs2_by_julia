include("../common/layers.jl") # Embedding
include("negative_sampling_layer.jl") # NegativeSamplingLoss


mutable struct CBOW
    in_layers
    ns_loss
    params
    grads
    word_vecs
end

function CBOW(vocab_size, hidden_size, window_size, corpus)
    V, H = vocab_size, hidden_size

    # 重みの初期化
    W_in = 0.01 * randn(Float32, V, H)
    W_out = 0.01 * randn(Float32, V, H)

    # レイヤの生成
    in_layers = [Embedding(W_in) for i=1:(2*window_size)] # Embeddingレイヤを使用

    ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

    # すべての重みと勾配をリストにまとめる
    layers = [in_layers..., ns_loss]
    params = typeof(layers[1].params)([])
    grads  = typeof(layers[1].grads)([])
    for layer in layers
        append!(params, layer.params)
        append!(grads,  layer.grads)
    end

    # メンバ変数に単語の分散表現を設定
    word_vecs = W_in

    return CBOW(in_layers, ns_loss, params, grads, word_vecs)
end

function forward(self::CBOW, contexts, target)
    h = 0
    for (i, layer) = enumerate(self.in_layers)
        h += forward(layer, selectdim(contexts, 2, i))
    end
    h /= length(self.in_layers)
    loss = forward(self.ns_loss, h, target)
    return loss
end

function backward(self::CBOW, dout=1)
    dout = backward(self.ns_loss, dout)
    dout /= length(self.in_layers)
    for layer = self.in_layers
        backward(layer, dout)
    end
end
