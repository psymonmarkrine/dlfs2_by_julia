function getpart(x::Array{T,N}, indexes) where {T,N}
    """Pythonのインデックスっぽく取り出すための関数
    ```
    julia> x = rand(10, 3, 28, 28);

    julia> size(x)
    (10, 3, 28, 28)

    julia> size(getpart(x, (2:5, :, 3)))
    (4, 3, 28)

    julia> size(getpart(x, (2:5,)))
    (4, 3, 28, 28)
    ```
    """
    l = length(indexes)
    d = ndims(x)
    return x[[i>l ? (:) : indexes[i] for i=1:d]...]
end

# function getpart2(x::Array{T,N}, indexes) where {T,N}
#     """Pythonのインデックスっぽく取り出すための関数
#       BenchmarkToolsで計測したところ上記getpartの方が早かったので没
#     """
#     x = copy(x)
#     l = length(indexes)
#     d = ndims(x)
#     for i=1:d
#         if i > l || indexes[i] == :
#             continue
#         else
#             x = selectdim(x, i, indexes[i])
#         end
#     end
#     return x
# end

function choice(num::Integer; p, with_index=false)
    return choice(1:num, p=p, with_index=with_index)
end

function choice(array; p, with_index=false)
    array = collect(array)
    l = min(length(array), length(p))
    r = rand()
    p = [sum(p[1:i]) for i=1:l]/sum(p[1:l])
    for i=1:l
        if p[i]>=r
            return with_index ? (array[i], i) : array[i]
        end
    end
end

function choice(num::Integer, size::Integer; replace=true, p)
    choice(1:num, size, replace=replace, p=p)
end

function choice(array, size::Integer; replace=true, p)
    if !replace
        array = collect(array)
        size = min(length(array), size)
        if size<=1
            return choice(array, p=p)
        end
        ret, idx = choice(array, p=p, with_index=true)
        popat!(array, idx)
        popat!(p, idx)
        return [ret, choice(array, size-1, p=p, replace=false)...]
    end
    return [choice(array, p=p) for _=1:size]
end

function choice(array, size::Tuple; replace=true, p) where N
    return reshape(choice(array, *(size...), replace=replace, p=p), size)
end

