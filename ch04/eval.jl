include("../common/util.jl") # most_similar, analogy
import HDF5


hdf_file = "cbow_params.h5"
# hdf_file = "skipgram_params.h5"

word_vecs  = HDF5.h5read(hdf_file, "word_vecs")
word_to_id = HDF5.h5read(hdf_file, "word_to_id")
id_to_word = HDF5.h5read(hdf_file, "id_to_word")

# most similar task
querys = ["you", "year", "car", "toyota"]
for query = querys
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
end

# analogy task
println("-"^50)
analogy("king", "man", "queen",  word_to_id, id_to_word, word_vecs)
analogy("take", "took", "go",  word_to_id, id_to_word, word_vecs)
analogy("car", "cars", "child",  word_to_id, id_to_word, word_vecs)
analogy("good", "better", "bad",  word_to_id, id_to_word, word_vecs)