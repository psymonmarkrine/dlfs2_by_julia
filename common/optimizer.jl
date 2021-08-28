

mutable struct SGD
    lr
    SGD(lr=0.01) = new(lr)
end

"""
確率的勾配降下法（Stochastic Gradient Descent）
"""
function update(self::SGD, params, grads)
    for i = 1:length(params)
        @. params[i] -= self.lr * grads[i]
    end
end


mutable struct Momentum
    lr
    momentum
    v
    Momentum(lr=0.01, momentum=0.9) = new(lr, momentum, nothing)
end

"""
Momentum SGD
"""
function update(self::Momentum, params, grads)
    if isnothing(self.v)
        self.v = [zero(i) for i=params]
    end
    for i = 1:length(params)
        @. self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
        @. params[i] += self.v[i]
    end
end


mutable struct Nesterov
    lr
    momentum
    v
    Nesterov(lr=0.01, momentum=0.9) = new(lr, momentum, nothing)
end

"""
Nesterov"s Accelerated Gradient (http://arxiv.org/abs/1212.0901)
"""
function update(self::Nesterov, params, grads)
    if isnothing(self.v)
        self.v = [zero(i) for i=params]
    end

    for i = 1:length(params)
        @. self.v[i] *= self.momentum
        @. self.v[i] -= self.lr * grads[i]
        @. params[i] += self.momentum ^ 2 * self.v[i]
        @. params[i] -= (1 + self.momentum) * self.lr * grads[i]
    end
end


mutable struct AdaGrad
    lr
    h
    AdaGrad(lr=0.01) = new(lr, nothing)
end
"""
AdaGrad
"""
function update(self::AdaGrad, params, grads)
    if isnothing(self.h)
        self.h = [zero(i) for i=params]
    end

    for i = 1:length(params)
        @. self.h[i] += grads[i] ^ 2
        @. params[i] -= self.lr * grads[i] / (sqrt(self.h[i]) + 1e-7)
    end
end

mutable struct RMSprop
    lr
    decay_rate
    h
    RMSprop(lr=0.01, decay_rate=0.99) = new(lr,decay_rate,nothing)
end
"""
RMSprop
"""
function update(self::RMSprop, params, grads)
    if isnothing(self.h)
        self.h = [zero(i) for i=params]
    end

    for i = 1:length(params)
        @. self.h[i] *= self.decay_rate
        @. self.h[i] += (1 - self.decay_rate) * grads[i] ^2
        @. params[i] -= self.lr * grads[i] / (sqrt(self.h[i]) + 1e-7)
    end
end


mutable struct Adam
    lr
    beta1
    beta2
    iter
    m
    v
    Adam(lr=0.001, beta1=0.9, beta2=0.999) = new(lr, beta1, beta2, 0, nothing, nothing)
end
"""
Adam (http://arxiv.org/abs/1412.6980v8)
"""
function update(self::Adam, params, grads)
    if isnothing(self.m)
        self.m = [zero(i) for i=params]
        self.v = [zero(i) for i=params]
    end

    self.iter += 1
    lr_t = self.lr * sqrt(1.0 - self.beta2^self.iter) / (1.0 - self.beta1^self.iter)

    for i = 1:length(params)
        @. self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
        @. self.v[i] += (1 - self.beta2) * (grads[i]^2 - self.v[i])
        
        @. params[i] -= lr_t * self.m[i] / (sqrt(self.v[i]) + 1e-7)
    end
end
