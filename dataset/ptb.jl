import Downloads
import JLD2


url_base = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
key_file = Dict(
    "train" => "ptb.train.txt",
    "test"  => "ptb.test.txt",
    "valid" => "ptb.valid.txt"
)
save_file = Dict(
    "train" => "ptb.train.jld2",
    "test"  => "ptb.test.jld2",
    "valid" => "ptb.valid.jld2"
)
vocab_file = "ptb.vocab.jld2"

dataset_dir = dirname(abspath(@__FILE__))


function _download(file_name)
    file_path = joinpath(dataset_dir, file_name)

    if isfile(file_path)
        return
    end

    println("Downloading $file_name ... ")

    Downloads.download(url_base * file_name, file_path)
    println("Done")
end

function load_vocab()
    vocab_path = joinpath(dataset_dir, vocab_file)

    if isfile(vocab_path)
        d = JLD2.load(vocab_path)
        return d["word_to_id"], d["id_to_word"]
    end

    word_to_id = Dict{String, Int16}()
    id_to_word = Dict{Int16, String}()
    data_type = "train"
    file_name = key_file[data_type]
    file_path = joinpath(dataset_dir, file_name)

    _download(file_name)

    words = open(file_path, "r") do f
        String.(split(strip(replace(String(read(f)), "\n"=>"<eos>"))))
    end

    for (i, word) = enumerate(words)
        if !(word in keys(word_to_id))
            tmp_id = length(word_to_id) + 1
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word
        end
    end

    JLD2.save(vocab_path, "id_to_word", id_to_word, "word_to_id", word_to_id)

    return word_to_id, id_to_word
end

function load_data(data_type="train")
    """
        :param data_type: データの種類："train" or "test" or "valid (val)"
        :return:
    """
    save_path = joinpath(dataset_dir, save_file[data_type])

    word_to_id, id_to_word = load_vocab()

    if isfile(save_path)
        corpus = JLD2.load(save_path, "corpus")
        return corpus, word_to_id, id_to_word
    end

    file_name = key_file[data_type]
    file_path = joinpath(dataset_dir, file_name)
    _download(file_name)

    words = open(file_path, "r") do f
        String.(split(strip(replace(String(read(f)), "\n"=>"<eos>"))))
    end
    corpus = [word_to_id[w] for w = words]
    
    JLD2.save(save_path, "corpus", corpus)
    return corpus, word_to_id, id_to_word
end

if abspath(PROGRAM_FILE) == @__FILE__
    for data_type = ("train", "valid", "test")
        load_data(data_type)
    end
end
