import Printf: @sprintf

GPU = false

function preprocess(text)
    text = lowercase(text)
    text = replace(text, "." => " .")
    words = split(text, " ")

    word_to_id = Dict{AbstractString, Integer}()
    id_to_word = Dict{Integer, AbstractString}()
    for word = words
        new_id = get!(word_to_id, word) do
            length(word_to_id) + 1
        end
        get!(id_to_word, new_id, word)
    end
    corpus = [word_to_id[w] for w = words]

    return corpus, word_to_id, id_to_word
end

function cos_similarity(x, y, eps=1e-8)
    """コサイン類似度の算出
    :param x: ベクトル
    :param y: ベクトル
    :param eps: ”0割り”防止のための微小値
    :return:
    """
    nx = x / (sqrt(sum(x .^ 2)) + eps)
    ny = y / (sqrt(sum(y .^ 2)) + eps)
    return nx' * ny
end

function most_similar(query, word_to_id, id_to_word, word_matrix; top=5)
    """類似単語の検索
    :param query: クエリ（テキスト）
    :param word_to_id: 単語から単語IDへのディクショナリ
    :param id_to_word: 単語IDから単語へのディクショナリ
    :param word_matrix: 単語ベクトルをまとめた行列。各行に対応する単語のベクトルが格納されていることを想定する
    :param top: 上位何位まで表示するか
    """
    if !(query in keys(word_to_id))
        error("$query is not found")
        return
    end

    println("\n[query] $query")
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id,:]

    vocab_size = length(id_to_word)

    similarity = zeros(vocab_size)
    for i = 1:vocab_size
        similarity[i] = cos_similarity(word_matrix[i,:], query_vec)
    end

    count = 0
    for i = sortperm(-similarity)
        if id_to_word[i] == query
            continue
        end
        println(" $(id_to_word[i]): $(similarity[i])")

        count += 1
        if count >= top
            return
        end
    end
end


function convert_one_hot(corpus::Vector{T}, vocab_size) where T
    """one-hot表現への変換
    :param corpus: 単語IDのリスト（1次元のNumPy配列）
    :param vocab_size: 語彙数
    :return: one-hot表現（2次元のNumPy配列）
    """
    N = size(corpus, 1)

    one_hot = zeros(Int32, (N, vocab_size))
    for (idx, word_id) = enumerate(corpus)
        one_hot[idx, word_id] = 1
    end
    return one_hot
end

function convert_one_hot(corpus::Matrix{T}, vocab_size) where T
    """one-hot表現への変換
    :param corpus: 単語IDのリスト（2次元のNumPy配列）
    :param vocab_size: 語彙数
    :return: one-hot表現（3次元のNumPy配列）
    """
    N, C = size(corpus)

    one_hot = zeros(Int32, (N, C, vocab_size))
    for idx_0 = 1:N
        for idx_1 = 1:C
            word_id = corpus[idx_0, idx_1]
            one_hot[idx_0, idx_1, word_id] = 1
        end
    end
    return one_hot
end


function create_co_matrix(corpus, vocab_size; window_size=1)
    """共起行列の作成
    :param corpus: コーパス（単語IDのリスト）
    :param vocab_size:語彙数
    :param window_size:ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
    :return: 共起行列
    """
    corpus_size = length(corpus)
    co_matrix = zeros(Int32, (vocab_size, vocab_size))

    for (idx, word_id) = enumerate(corpus)
        for i = 1:window_size
            left_idx = idx - i
            right_idx = idx + i

            if left_idx > 0
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            end
            if right_idx <= corpus_size
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
            end
        end
    end
    return co_matrix
end


function ppmi(C; verbose=false, eps = 1e-8)
    """PPMI（正の相互情報量）の作成
    :param C: 共起行列
    :param verbose: 進行状況を出力するかどうか
    :return:
    """
    M = zeros(Float32, size(C))
    N = sum(C)
    S = sum(C, dims=1)
    total = size(C, 1) * size(C, 2)
    cnt = 0

    for i = 1:size(C, 1)
        for j = 1:size(C, 2)
            pmi = log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose
                cnt += 1
                if cnt % (div(total, 100) + 1) == 0
                    println(@sprintf("%.1f%% done", 100*cnt/total))
                end
            end
        end
    end
    return M
end


function create_contexts_target(corpus; window_size=1)
    """コンテキストとターゲットの作成
    :param corpus: コーパス（単語IDのリスト）
    :param window_size: ウィンドウサイズ（ウィンドウサイズが1のときは、単語の左右1単語がコンテキスト）
    :return:
    """
    target = corpus[(window_size+1):(end-window_size)]
    contexts = zeros(eltype(corpus), (0, 2window_size))

    for idx = (window_size+1):(length(corpus)-window_size)
        cs = zeros(eltype(corpus), (1, 0))
        for t = -window_size:window_size
            if t == 0
                continue
            end
            cs = hcat(cs, corpus[idx + t])
        end
        contexts = vcat(contexts, cs)
    end
    return contexts, target
end

#=
def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)

# =#
function clip_grads(grads, max_norm)
    total_norm = sqrt(sum([sum(grad.^2) for grad=grads]))
    
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1
        for grad = grads
            grad .*= rate
        end
    end

    return
end

#=
def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print("evaluating perplexity ...")
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write("\r%d / %d" % (iters, max_iters))
        sys.stdout.flush()

    print("")
    ppl = np.exp(total_loss / max_iters)
    return ppl


def eval_seq2seq(model, question, correct, id_to_char,
                 verbos=False, is_reverse=False):
    correct = correct.flatten()
    # 頭の区切り文字
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 文字列へ変換
    question = "".join([id_to_char[int(c)] for c in question.flatten()])
    correct = "".join([id_to_char[int(c)] for c in correct])
    guess = "".join([id_to_char[int(c)] for c in guess])

    if verbos:
        if is_reverse:
            question = question[::-1]

        colors = {"ok": "\033[92m", "fail": "\033[91m", "close": "\033[0m"}
        print("Q", question)
        print("T", correct)

        is_windows = os.name == "nt"

        if correct == guess:
            mark = colors["ok"] + "☑" + colors["close"]
            if is_windows:
                mark = "O"
            print(mark + " " + guess)
        else:
            mark = colors["fail"] + "☒" + colors["close"]
            if is_windows:
                mark = "X"
            print(mark + " " + guess)
        print("---")

    return 1 if guess == correct else 0


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print("%s is not found" % word)
            return

    print("\n[analogy] " + a + ":" + b + " = " + c + ":?")
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(" {0}: {1}".format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x

# =#