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


