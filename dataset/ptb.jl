import Downloads
import HDF5


url_base = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/"
key_file = Dict(
    "train" => "ptb.train.txt",
    "test"  => "ptb.test.txt",
    "valid" => "ptb.valid.txt"
)
save_file = "ptb.h5"
vocab_file = "ptb.vocab.h5"

dataset_dir = dirname(abspath(@__FILE__))


function _download(file_name)
    file_path = joinpath(dataset_dir, file_name)

    if isfile(file_path)
        return
    end

    println("Downloading $file_name ... ")

    # headers = Dict("User-Agent" => "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0")
    Downloads.download(url_base * file_name, file_path)
    println("Done")
end

function load_vocab()
    vocab_path = joinpath(dataset_dir, vocab_file)

    if isfile(vocab_path)
        word_to_id = Dict{AbstractString, Integer}()
        id_to_word = Dict{Integer, AbstractString}()
        for i = 1:1000000
            try
                id_to_word[i] = HDF5.h5read(save_file, "$i")
                word_to_id[id_to_word[i]] = i
            catch
                println(i)
                break
            end
        end
        return word_to_id, id_to_word
    end

    word_to_id = Dict{AbstractString, Integer}()
    id_to_word = Dict{Integer, AbstractString}()
    data_type = "train"
    file_name = key_file[data_type]
    file_path = joinpath(dataset_dir, file_name)

    _download(file_name)

    # words = open(file_path).read().replace("\n", "<eos>").strip().split()
    words = open(file_path, "r") do f
        split(strip(replace(String(read(f)), "\n"=>"<eos>")))
    end

    for (i, word) = enumerate(words)
        if !(word in values(word_to_id))
            tmp_id = length(word_to_id) + 1
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word
        end
    end

    for (k, v) = id_to_word
        HDF5.h5write(vocab_path, "$k", v)
    end

    return word_to_id, id_to_word
end

function load_data(data_type="train")
    """
        :param data_type: データの種類："train" or "test" or "valid (val)"
        :return:
    """
    save_path = joinpath(dataset_dir, save_file)

    word_to_id, id_to_word = load_vocab()

    try
        corpus = HDF5.h5read(save_path, data_type)
        return corpus, word_to_id, id_to_word
    catch
        ;
    end

    file_name = key_file[data_type]
    file_path = joinpath(dataset_dir, file_name)
    _download(file_name)

    words = open(file_path, "r") do f
        split(strip(replace(String(read(f)), "\n"=>"<eos>")))
    end
    corpus = [word_to_id[w] for w = words]
    
    HDF5.h5write(save_path, data_type, corpus)
    return corpus, word_to_id, id_to_word
end

if abspath(PROGRAM_FILE) == @__FILE__
    for data_type = ("train", "valid", "test")
        load_data(data_type)
    end
end
