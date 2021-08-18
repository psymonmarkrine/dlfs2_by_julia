import Random: seed!

function load_data(seed=1984)
    seed!(seed)
    N = 100  # クラスごとのサンプル数
    DIM = 2  # データの要素数
    CLS_NUM = 3  # クラス数

    x = zeros(N*CLS_NUM, DIM)
    t = zeros(Integer, N*CLS_NUM, CLS_NUM)

    for j = 0:CLS_NUM-1
        for i = 1:N #N*j, N*(j+1))
            rate = i / N
            radius = rate
            theta = 4j + 4rate + randn()*0.2

            ix = N*j + i
            x[ix, :] .= (radius*sin(theta), radius*cos(theta))
            t[ix, j+1] = 1
        end
    end
    return x, t
end