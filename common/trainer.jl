import Dates: now
import Random: randperm
import Printf: @sprintf

using Plots
include("util.jl") # clip_grads
include("python_likes.jl")

GPU = false


mutable struct Trainer
    model
    optimizer
    loss_list
    eval_interval
    current_epoch
    Trainer(model, optimizer) = new(model, optimizer, zeros(0), nothing, 0)
end

function fit(self::Trainer, x, t, max_epoch=10, batch_size=32; max_grad=nothing, eval_interval=20)
    data_size = size(x, 1)
    max_iters = div(data_size, batch_size)
    self.eval_interval = eval_interval
    model, optimizer = self.model, self.optimizer
    total_loss = 0
    loss_count = 0

    start_time = now()
    for epoch = 1:max_epoch
        # シャッフル
        idx = randperm(data_size)
        x = getpart(x, [idx])
        t = getpart(t, [idx])

        self.current_epoch += 1
        for iters = 1:max_iters
            idx_end = iters * batch_size
            idx_begin = idx_end - batch_size + 1
            batch_x = getpart(x, [idx_begin:idx_end])
            batch_t = getpart(t, [idx_begin:idx_end])
            
            # 勾配を求め、パラメータを更新
            loss = forward(model, batch_x, batch_t)
            backward(model)
            params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
            if !isnothing(max_grad)
                clip_grads(grads, max_grad)
            end
            update(optimizer, params, grads)
            total_loss += loss
            loss_count += 1

            # 評価
            if !isnothing(eval_interval) && (iters % eval_interval) == 1
                avg_loss = total_loss / loss_count
                elapsed_time = now() - start_time
                println("| epoch $(self.current_epoch) |  iter $iters / $max_iters | time $elapsed_time | loss $(@sprintf("%.2f", avg_loss))")
                append!(self.loss_list, avg_loss)
                total_loss, loss_count = 0, 0
            end
        end
    end
end

function plot(self::Trainer; ylim=nothing)
    x = 1:length(self.loss_list)
    plot!(xlabel="iterations (x $(self.eval_interval))", ylabel="loss")
    if !isnothing(ylim)
        plot!(ylim=ylim)
    end
    plot!(x, self.loss_list, label="train")
end

#=
class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # バッチの各サンプルの読み込み開始位置

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # パープレキシティの評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print("| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f"
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label="train")
        plt.xlabel("iterations (x" + str(self.eval_interval) + ")")
        plt.ylabel("perplexity")
        plt.show()

# =#
function remove_duplicate(params, grads)
    """
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    """
    params, grads = copy(params), copy(grads)  # copy list
    
    while true
        find_flg = false
        L = size(params, 1)

        for i = 1:(L-1)
            for j = (i+1):L
                # 重みを共有する場合
                if params[i] === params[j]
                    grads[i] .+= grads[j]  # 勾配の加算
                    find_flg = true
                    popat!(params, j)
                    popat!(grads, j)
                # 転置行列として重みを共有する場合（weight tying）
                elseif ndims(params[i])==2 && ndims(params[j])==2 && size(params[i]')==size(params[j]) && all(params[i]' .== params[j])
                    grads[i] .+= grads[j]'
                    find_flg = true
                    popat!(params, j)
                    popat!(grads, j)
                end
                if find_flg
                    break
                end
            end
            if find_flg
                break
            end
        end
        if !find_flg
            break
        end
    end

    return params, grads
end
