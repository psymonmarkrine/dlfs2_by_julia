import Printf: @printf

include("../common/util.jl") # preprocess, create_co_matrix, cos_similarity, ppmi


text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = length(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

Base.show(io::IO, f::Float32) = @printf(io, "%.3f", f) # 有効桁３桁で表示
println("covariance matrix")
display(C)
println("-"^50)
println("PPMI")
display(W)
