<<<<<<< HEAD
% MNISTデータ読み込み
[XTrain, YTrain] = digitTrain4DArrayData;

% 「0」,「1」のみ
idx = (YTrain == '0') | (YTrain == '1');
X = reshape(XTrain(:,:,:,idx), [], sum(idx))';
Y = double(YTrain(idx) == '1'); % 0/1ラベル

% クライアントごとにIIDに分割
N = size(X,1);
% Nの1行目の大きさ（つまり画像数）
K = 30;
perm = randperm(N);
X = X(perm,:);
Y = Y(perm);
data_per_client = floor(N/K);

for k = 1:K
    idk = (k-1)*data_per_client + 1 : k*data_per_client;
    X_clients{k} = X(idk,:);%セル配列
    Y_clients{k} = Y(idk);
end

alpha = 1;  % ステップサイズ
mu = 0.01;  % 学習率

% 攻撃者を20%に設定<-変更する
attackers = randperm(K, floor(K/5));

num_epochs = 500;
attack_detect_count = zeros(1,K);  % 各クライアントが攻撃者と判定された回数

for epoch = 1:num_epochs
    grads = zeros(size(X,2), K);  % 毎エポック初期化

    for k = 1:K
        Xk = X_clients{k};
        Yk = Y_clients{k};

        w_local = zeros(size(X,2),1); % 重み初期化
        pred = 1./(1+exp(-Xk*w_local));
        grad = Xk' * (pred - Yk) / size(Xk,1);

        if ismember(k, attackers)
            grad = -grad;  % sign-flipping attack
        end

        grads(:,k) = grad;
    end

    mean_grad = mean(grads, 2);  % 全体平均勾配
    cos_sims = (grads' * mean_grad) ./ ...
               (vecnorm(grads,2,1)' * norm(mean_grad));
    
    tau = -2;  % 本文の論理にあわせ調整
    detected = find(cos_sims < tau);  % コサイン類似度で判定
    attack_detect_count(detected) = attack_detect_count(detected) + 1;  % カウント
end


% グローバル平均勾配
mean_grad = mean(grads, 2);

valid_clients = setdiff(1:K, detected_attackers);
global_grad = mean(grads(:,valid_clients), 2);

w_global = zeros(size(global_grad)); % グローバル初期
w_global = w_global - mu * global_grad;

=======
% MNISTデータ読み込み
[XTrain, YTrain] = digitTrain4DArrayData;

% 「0」,「1」のみ
idx = (YTrain == '0') | (YTrain == '1');
X = reshape(XTrain(:,:,:,idx), [], sum(idx))';
Y = double(YTrain(idx) == '1'); % 0/1ラベル

% クライアントごとにIIDに分割
N = size(X,1);
% Nの1行目の大きさ（つまり画像数）
K = 30;
perm = randperm(N);
X = X(perm,:);
Y = Y(perm);
data_per_client = floor(N/K);

for k = 1:K
    idk = (k-1)*data_per_client + 1 : k*data_per_client;
    X_clients{k} = X(idk,:);%セル配列
    Y_clients{k} = Y(idk);
end

alpha = 1;  % ステップサイズ
mu = 0.01;  % 学習率

% 攻撃者を20%に設定<-変更する
attackers = randperm(K, floor(K/5));

num_epochs = 500;
attack_detect_count = zeros(1,K);  % 各クライアントが攻撃者と判定された回数

for epoch = 1:num_epochs
    grads = zeros(size(X,2), K);  % 毎エポック初期化

    for k = 1:K
        Xk = X_clients{k};
        Yk = Y_clients{k};

        w_local = zeros(size(X,2),1); % 重み初期化
        pred = 1./(1+exp(-Xk*w_local));
        grad = Xk' * (pred - Yk) / size(Xk,1);

        if ismember(k, attackers)
            grad = -grad;  % sign-flipping attack
        end

        grads(:,k) = grad;
    end

    mean_grad = mean(grads, 2);  % 全体平均勾配
    cos_sims = (grads' * mean_grad) ./ ...
               (vecnorm(grads,2,1)' * norm(mean_grad));
    
    tau = -2;  % 本文の論理にあわせ調整
    detected = find(cos_sims < tau);  % コサイン類似度で判定
    attack_detect_count(detected) = attack_detect_count(detected) + 1;  % カウント
end


% グローバル平均勾配
mean_grad = mean(grads, 2);

valid_clients = setdiff(1:K, detected_attackers);
global_grad = mean(grads(:,valid_clients), 2);

w_global = zeros(size(global_grad)); % グローバル初期
w_global = w_global - mu * global_grad;

