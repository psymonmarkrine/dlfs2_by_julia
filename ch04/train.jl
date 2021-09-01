# GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
# ===============================================
# config.GPU = True
# ===============================================
import HDF5
include("../common/trainer.jl") # Trainer
include("../common/optimizer.jl") # Adam
include("cbow.jl") # CBOW
include("skip_gram.jl") # SkipGram
include("../common/util.jl") # create_contexts_target, to_cpu, to_gpu
include("../dataset/ptb.jl")


# ハイパーパラメータの設定
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# データの読み込み
corpus, word_to_id, id_to_word = load_data("train")
vocab_size = length(word_to_id)

contexts, target = create_contexts_target(corpus, window_size=window_size)
# if config.GPU:
#     contexts, target = to_gpu(contexts), to_gpu(target)
# end

# モデルなどの生成
model = CBOW(vocab_size, hidden_size, window_size, corpus)
# model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 学習開始
fit(trainer, contexts, target, max_epoch, batch_size)
plot(trainer)

# 後ほど利用できるように、必要なデータを保存
word_vecs = model.word_vecs
# if config.GPU:
#     word_vecs = to_cpu(word_vecs)
# end
params = Dict(
    "word_vecs" => Float16.(word_vecs),
    "word_to_id" => word_to_id,
    "id_to_word" => id_to_word
)
h5_file = "cbow_params.h5"  # or "skipgram_params.h5"
for (k, v) = params
    HDF5.h5write(h5_file, k, v)
end
