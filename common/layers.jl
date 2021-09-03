# from common.np import *  # import numpy as np
# from common.config import GPU = false
include("functions.jl") # softmax, cross_entropy_error

GPU = false

mutable struct MatMul
    params
    grads
    x
end

MatMul(W) = MatMul([W], [zero(W)], nothing)

function forward(self::MatMul, x)
    W = self.params[1]
    out = x * W
    self.x = x
    return out
end

function backward(self::MatMul, dout)
    W = self.params[1]
    dx = dout * W'
    dW = self.x' * dout
    self.grads[1] .= dW
    return dx
end


mutable struct Affine
    params
    grads
    x
end

Affine(W, b) = Affine([W, b], [zero(W), zero(b)], nothing)

function forward(self::Affine, x)
    W, b = self.params
    out = x * W .+ b
    self.x = x
    return out
end

function backward(self::Affine, dout)
    W, b = self.params
    dx = dout * W'
    dW = self.x' * dout
    db = sum(dout, dims=1)

    self.grads[1] .= dW
    self.grads[2] .= db
    return dx
end


mutable struct Softmax
    params
    grads
    out
end

Softmax() = Softmax([], [], nothing)

function forward(self::Softmax, x)
    self.out = softmax(x)
    return self.out
end

function backward(self::Softmax, dout)
    dx = self.out .* dout
    sumdx = sum(dx, dims=2)
    dx -= self.out .* sumdx
    return dx
end


mutable struct SoftmaxWithLoss
    params
    grads
    y
    t
end

SoftmaxWithLoss() = SoftmaxWithLoss([], [], nothing, nothing)

function forward(self::SoftmaxWithLoss, x, t)
    self.t = t
    self.y = softmax(x)

    loss = cross_entropy_error(self.y, self.t)
    return loss
end

function backward(self::SoftmaxWithLoss, dout=1)
    batch_size = size(self.t, 1)
    t = [i[2] for i=argmax(self.t, dims=2)]
    dx = copy(self.y)
    for (i,t) = enumerate(t)
        dx[i, t] -= 1
    end
    dx .*= dout
    dx = dx / batch_size

    return dx
end


mutable struct Sigmoid
    params
    grads
    out
end

Sigmoid() = Sigmoid([], [], nothing)

function forward(self::Sigmoid, x)
    out = @. 1 / (1 + exp(-x))
    self.out = out
    return out
end

function backward(self::Sigmoid, dout)
    dx = @. dout * (1.0 - self.out) * self.out
    return dx
end


mutable struct SigmoidWithLoss
    params
    grads
    loss
    y
    t
end

SigmoidWithLoss() = SigmoidWithLoss([], [], nothing, nothing, nothing)

function forward(self::SigmoidWithLoss, x, t)
    self.t = t
    self.y = @. 1 / (1 + exp(-x))

    self.loss = cross_entropy_error(hcat(1 .- self.y, self.y), self.t)

    return self.loss
end

function backward(self::SigmoidWithLoss, dout=1)
    batch_size = size(self.t, 1)

    dx = (self.y - self.t) * dout / batch_size
    return dx
end


mutable struct Dropout
    params
    grads
    dropout_rate
    mask
end

"""
http://arxiv.org/abs/1207.0580
"""
Dropout(dropout_rate=0.5) = Dropout([], [], dropout_rate, nothing)

function forward(self::Dropout, x, train_flg=true)
    if train_flg
        self.mask = rand(size(x)) .> self.dropout_ratio
        return x .* self.mask
    else
        return x * (1.0 - self.dropout_ratio)
    end
end

function backward(self, dout)
    return dout .* self.mask
end


mutable struct Embedding
    params
    grads
    idx
end

Embedding(W) = Embedding([W], [zero(W)], nothing)

function forward(self::Embedding, idx)
    W = self.params[1]
    self.idx = idx
    out = selectdim(W, 1, idx)
    return out
end

function backward(self::Embedding, dout)
    dW = self.grads[1]
    dW .= 0.0
    if GPU
        np.scatter_add(dW, self.idx, dout)
    else
        selectdim(dW, 1, self.idx) .+= dout
    end
    return nothing
end
