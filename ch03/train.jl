include("../common/trainer.jl") # Trainer
include("../common/optimizer.jl") # Adam
include("simple_cbow.jl") # SimpleCBOW
include("../common/util.jl") # preprocess, create_contexts_target, convert_one_hot


window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = length(word_to_id)
contexts, target = create_contexts_target(corpus, window_size=window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

fit(trainer, contexts, target, max_epoch, batch_size)
plot(trainer)

word_vecs = model.word_vecs
for (word_id, word) = id_to_word
    println("$word $(word_vecs[word_id,:])")
end

    