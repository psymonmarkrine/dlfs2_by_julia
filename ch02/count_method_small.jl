import Printf: @printf
import LinearAlgebra: svd

using Plots
include("../common/util.jl") # preprocess, create_co_matrix, ppmi


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = length(id_to_word)
C = create_co_matrix(corpus, vocab_size, window_size=1)
W = ppmi(C)

# SVD
U, S, V = svd(W)

Base.show(io::IO, f::Float32) = @printf(io, "%.3f", f) # 有効桁３桁で表示
println(C[1,:])
println(W[1,:])
println(U[1,:])

# plot
scatter(U[:,1], U[:,2], leg=false)
annotate!([(U[word_id, 1], U[word_id, 2], word) for (word, word_id) = word_to_id]..., annotationhalign=:left)
savefig("../image/ch02/fig02-11.png")
