import LinearAlgebra: svd

include("../common/util.jl") # most_similar, create_co_matrix, ppmi
include("../dataset/ptb.jl")


window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = load_data("train")
vocab_size = length(word_to_id)
println("counting  co-occurrence ...")
C = create_co_matrix(corpus, vocab_size, window_size=window_size)
println("calculating PPMI ...")
W = ppmi(C, verbose=true)

print("calculating SVD ...")
U, S, V = svd(W)

word_vecs = U[:, 1:wordvec_size]

querys = ["you", "year", "car", "toyota"]
for query in querys
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
end
