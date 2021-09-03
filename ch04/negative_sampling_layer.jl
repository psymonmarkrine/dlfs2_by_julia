include("../common/layers.jl") # Embedding, SigmoidWithLoss
include("../common/python_likes.jl") # choice
# import collections


mutable struct EmbeddingDot
    embed
    params
    grads
    cache
end

function EmbeddingDot(W)
    embed = Embedding(W)
    params = embed.params
    grads = embed.grads
    return EmbeddingDot(embed, params, grads, nothing)
end

function forward(self::EmbeddingDot, h, idx)
    target_W = forward(self.embed, idx)
    out = sum(target_W * h, dims=2)

    self.cache = (h, target_W)
    return out
end

function backward(self::EmbeddingDot, dout)
    h, target_W = self.cache
    dout = reshape(dout, size(dout, 1), 1)

    dtarget_W = dout .* h
    backward(self.embed, dtarget_W)
    dh = dout .* target_W
    return dh
end


mutable struct UnigramSampler
    sample_size::Integer
    vocab_size::Integer
    word_p::Vector
end

function UnigramSampler(corpus, power::AbstractFloat, sample_size::Integer)
    counts = Dict()
    for word_id = corpus
        get!(counts, word_id, 0)
        counts[word_id] += 1
    end

    vocab_size = length(counts)

    word_p = zeros(vocab_size)
    for i = 1:vocab_size
        word_p[i] = get(counts, i, 0)
    end
    word_p .^= power
    word_p ./= sum(word_p)

    return UnigramSampler(sample_size, vocab_size, word_p)
end

function get_negative_sample(self::UnigramSampler, target)
    batch_size = size(target, 1)

    # if !GPU
    negative_sample = zeros(Int32, (batch_size, self.sample_size))

    for i = 1:batch_size
        p = copy(self.word_p)
        target_idx = target[i]
        p[target_idx] = 0
        p /= sum(p)
        negative_sample[i, :] = choice(self.vocab_size, self.sample_size, replace=false, p=p)
    end
    # else
    #     # GPU(cupy）で計算するときは、速度を優先
    #     # 負例にターゲットが含まれるケースがある
    #     negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
    #                                         replace=True, p=self.word_p)
    # end
    return negative_sample
end


mutable struct NegativeSamplingLoss
    sample_size
    sampler
    loss_layers
    embed_dot_layers
    params
    grads
end

function NegativeSamplingLoss(W, corpus; power=0.75, sample_size=5)
    sample_size = sample_size
    sampler = UnigramSampler(corpus, power, sample_size)
    loss_layers = [SigmoidWithLoss() for _ = 0:sample_size]
    embed_dot_layers = [EmbeddingDot(W) for _ = 0:sample_size]

    params = typeof(embed_dot_layers[1].params)([])
    grads  = typeof(embed_dot_layers[1].grads)([])
    for layer = embed_dot_layers
        append!(params, layer.params)
        append!(grads,  layer.grads)
    end

    return NegativeSamplingLoss(sample_size, sampler, loss_layers, embed_dot_layers, params, grads)
end

function forward(self::NegativeSamplingLoss, h, target)
    batch_size = size(target, 1)
    negative_sample = get_negative_sample(self.sampler, target)

    # 正例のフォワード
    score = forward(self.embed_dot_layers[1], h, target)
    correct_label = ones(Int32, batch_size)
    loss = forward(self.loss_layers[1], score, correct_label)

    # 負例のフォワード
    negative_label = ones(Int32, batch_size)
    for i = 1:self.sample_size
        negative_target = selectdim(negative_sample, 2, i)
        score = forward(self.embed_dot_layers[1 + i], h, negative_target)
        loss += forward(self.loss_layers[1 + i], score, negative_label)
    end
    return loss
end

function backward(self::NegativeSamplingLoss, dout=1)
    # dh = 0
    # for (l0, l1) = zip(self.loss_layers, self.embed_dot_layers)
    #     dscore = backward(l0, dout)
    #     dh .+= backward(l1, dscore)
    # end
    dh = sum([backward(l1, backward(l0, dout)) for (l0, l1) = zip(self.loss_layers, self.embed_dot_layers)])
    return dh
end
