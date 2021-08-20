include("../common/optimizer.jl") # SGD
include("../common/trainer.jl") # Trainer
include("../dataset/spiral.jl")
include("two_layer_net.jl") # TwoLayerNet


# ハイパーパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = load_data()
model = TwoLayerNet(2, hidden_size, 3)
optimizer = SGD(learning_rate)

trainer = Trainer(model, optimizer)
fit(trainer, x, t, max_epoch, batch_size, eval_interval=10)
plot(trainer)
