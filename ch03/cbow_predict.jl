include("../common/layers.jl") # MatMul


# サンプルのコンテキストデータ
c0 = [1 0 0 0 0 0 0]
c1 = [0 0 1 0 0 0 0]

# 重みの初期化
W_in = randn(7, 3)
W_out = randn(3, 7)

# レイヤの生成
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 順伝搬
h0 = forward(in_layer0, c0)
h1 = forward(in_layer1, c1)
h = 0.5 * (h0 + h1)
s = forward(out_layer, h)
println(s)
