include("../dataset/ptb.jl")


corpus, word_to_id, id_to_word = load_data("train")

println("corpus size:", length(corpus))
println("corpus[1:30]:", corpus[1:30])
println()
println("id_to_word[1]:", id_to_word[1])
println("id_to_word[2]:", id_to_word[2])
println("id_to_word[3]:", id_to_word[3])
println()
println("word_to_id[\"car\"]:", word_to_id["car"])
println("word_to_id[\"happy\"]:", word_to_id["happy"])
println("word_to_id[\"lexus\"]:", word_to_id["lexus"])
