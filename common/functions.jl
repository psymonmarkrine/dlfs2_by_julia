# from common.np import *


function sigmoid(x)
    return 1 / (1 + exp(-x))
end


function relu(x)
    return max(0, x)
end


function softmax(x::Array{T,1}) where T
    x = x - maximum(x)
    x = exp.(x) ./ sum(exp.(x))
    return x
end

function softmax(x::Array{T,2}) where T
    x = x .- maximum(x, dims=2)
    x = exp.(x)
    x ./= sum(x, dims=2)
    return x
end


function cross_entropy_error(y::Array{T, 1}, t) where T
    t = reshape(t, 1, :)
    y = reshape(y, 1, :)

    return cross_entropy_error(y, t)
end

function cross_entropy_error(y, t)
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if lenth(t) == length(y)
        t = [i[2] for i=argmax(t, dims=2)]
    end
    batch_size = size(y, 1)

    return -sum(log.([y[i, t] + 1e-7 for (i,t)=enumerate(t)])) / batch_size
end
