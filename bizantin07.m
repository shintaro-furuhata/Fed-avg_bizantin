<<<<<<< HEAD
% =========================
% ラベルを変更することで攻撃 
% =========================

% =========================
% パラメータ設定
% =========================
K = 30;         % クライアント数
mu = 0.01;      % 学習率
tau = -2;       % 提案手法の閾値
epochs = 1000;  % エポック数
detect_epochs = 500; % 攻撃検出フェーズ
acc_threshold_ratio = 0.98; % 検出基準（割合）
attacker_ratio = 1/3; % 攻撃者割合

% =========================
% MNISTデータ読み込み (0と1のみ)
% =========================
[XTrain, YTrain] = digitTrain4DArrayData;
idx = (YTrain == '0') | (YTrain == '1');
X = reshape(XTrain(:,:,:,idx), [], sum(idx))';
Y = double(YTrain(idx) == '1');

% =========================
% 条件A/B/Cのループ
% =========================
conditions = {'A','B','C'};
results_acc = cell(1, length(conditions));

for cond_idx = 1:length(conditions)
    cond = conditions{cond_idx};
    fprintf("\n=== 条件%s 開始 ===\n", cond);

    % IID分割
    N = size(X,1);
    perm = randperm(N);
    X_perm = X(perm,:);
    Y_perm = Y(perm);
    data_per_client = floor(N/K);
    X_clients = cell(1,K); Y_clients = cell(1,K);
    for k = 1:K
        idk = (k-1)*data_per_client + 1 : k*data_per_client;
        X_clients{k} = X_perm(idk,:);
        Y_clients{k} = Y_perm(idk);
    end

    % 攻撃者設定
    attackers = randperm(K, floor(K * attacker_ratio));
    fprintf("実際の攻撃者: %s\n", mat2str(attackers));
    fprintf("正規クライアント: %s\n", mat2str(setdiff(1:K, attackers)));

    % 重み初期化
    w_global = zeros(size(X,2),1);
    detected_attackers = [];
    suspect_counts = zeros(K,1); % 条件C用

    acc_history = zeros(1, epochs);

    for epoch = 1:epochs
        grads = zeros(size(X,2), K);
        acc_clients = zeros(1, K);

        for k = 1:K
            Xk = X_clients{k};
            Yk = Y_clients{k};  % 元ラベルを取得

            % --- 攻撃者ラベル反転（条件B/C） ---
            if ismember(cond, ["B","C"]) && ismember(k, attackers)
                if cond == "B" || (cond == "C" && epoch <= detect_epochs)
                    Yk = 1 - Yk;  % 攻撃者は反転
                end
            end

            % 精度計算
            pred_labels = double((Xk * w_global) >= 0);
            acc_clients(k) = mean(pred_labels == Yk) * 100;

            % 勾配計算
            pred = 1 ./ (1 + exp(-Xk * w_global));
            grad = Xk' * (pred - Yk) / size(Xk,1);

            % 条件C: 検出後はラベルを元に戻す
            if cond == "C" && ismember(k, detected_attackers) && epoch > detect_epochs
                Yk = 1 - Yk;      % ラベル正常化
                grad = Xk' * (pred - Yk) / size(Xk,1); % 勾配も再計算
            end

            grads(:,k) = grad;
        end

        % 条件C: 攻撃検出処理
        if cond == "C" && epoch <= detect_epochs
            cos_sims = zeros(K,K);
            for i = 1:K
                for j = 1:K
                    if i ~= j
                        g_i = grads(:,i);
                        g_j = grads(:,j);
                        if norm(g_i) ~= 0 && norm(g_j) ~= 0
                            cos_sims(i,j) = (g_i' * g_j) / (norm(g_i) * norm(g_j));
                        end
                    end
                end
            end
            for i = 1:K
                sum_cos = sum(cos_sims(i,:));
                if sum_cos <= tau
                    suspect_counts(i) = suspect_counts(i) + 1;
                end
            end
            if epoch == detect_epochs
                detected_attackers = find(suspect_counts >= acc_threshold_ratio * detect_epochs);
                fprintf("条件C 検出された攻撃者: %s\n", mat2str(detected_attackers));
            end
        end

        % 勾配集約・モデル更新
        global_grad = mean(grads, 2);
        w_global = w_global - mu * global_grad;

        % 記録
        acc_history(epoch) = mean(acc_clients);

        % ログ
        if mod(epoch,100) == 0 || epoch == epochs
            fprintf("条件%s Epoch %d: 精度 %.2f%%\n", cond, epoch, acc_history(epoch));
        end
    end

    results_acc{cond_idx} = acc_history;
end

% =========================
% グラフ描画
% =========================
figure; hold on;
colors = {'b','r','g'};
for i = 1:length(conditions)
    plot(1:epochs, results_acc{i}, 'LineWidth', 1.5, 'Color', colors{i});
end
xlabel('Epoch'); ylabel('Accuracy (%)');
legend(conditions); title('精度推移'); grid on;
=======
% =========================
% ラベルを変更することで攻撃 
% =========================

% =========================
% パラメータ設定
% =========================
K = 30;         % クライアント数
mu = 0.01;      % 学習率
tau = -2;       % 提案手法の閾値
epochs = 1000;  % エポック数
detect_epochs = 500; % 攻撃検出フェーズ
acc_threshold_ratio = 0.98; % 検出基準（割合）
attacker_ratio = 1/3; % 攻撃者割合

% =========================
% MNISTデータ読み込み (0と1のみ)
% =========================
[XTrain, YTrain] = digitTrain4DArrayData;
idx = (YTrain == '0') | (YTrain == '1');
X = reshape(XTrain(:,:,:,idx), [], sum(idx))';
Y = double(YTrain(idx) == '1');

% =========================
% 条件A/B/Cのループ
% =========================
conditions = {'A','B','C'};
results_acc = cell(1, length(conditions));

for cond_idx = 1:length(conditions)
    cond = conditions{cond_idx};
    fprintf("\n=== 条件%s 開始 ===\n", cond);

    % IID分割
    N = size(X,1);
    perm = randperm(N);
    X_perm = X(perm,:);
    Y_perm = Y(perm);
    data_per_client = floor(N/K);
    X_clients = cell(1,K); Y_clients = cell(1,K);
    for k = 1:K
        idk = (k-1)*data_per_client + 1 : k*data_per_client;
        X_clients{k} = X_perm(idk,:);
        Y_clients{k} = Y_perm(idk);
    end

    % 攻撃者設定
    attackers = randperm(K, floor(K * attacker_ratio));
    fprintf("実際の攻撃者: %s\n", mat2str(attackers));
    fprintf("正規クライアント: %s\n", mat2str(setdiff(1:K, attackers)));

    % 重み初期化
    w_global = zeros(size(X,2),1);
    detected_attackers = [];
    suspect_counts = zeros(K,1); % 条件C用

    acc_history = zeros(1, epochs);

    for epoch = 1:epochs
        grads = zeros(size(X,2), K);
        acc_clients = zeros(1, K);

        for k = 1:K
            Xk = X_clients{k};
            Yk = Y_clients{k};  % 元ラベルを取得

            % --- 攻撃者ラベル反転（条件B/C） ---
            if ismember(cond, ["B","C"]) && ismember(k, attackers)
                if cond == "B" || (cond == "C" && epoch <= detect_epochs)
                    Yk = 1 - Yk;  % 攻撃者は反転
                end
            end

            % 精度計算
            pred_labels = double((Xk * w_global) >= 0);
            acc_clients(k) = mean(pred_labels == Yk) * 100;

            % 勾配計算
            pred = 1 ./ (1 + exp(-Xk * w_global));
            grad = Xk' * (pred - Yk) / size(Xk,1);

            % 条件C: 検出後はラベルを元に戻す
            if cond == "C" && ismember(k, detected_attackers) && epoch > detect_epochs
                Yk = 1 - Yk;      % ラベル正常化
                grad = Xk' * (pred - Yk) / size(Xk,1); % 勾配も再計算
            end

            grads(:,k) = grad;
        end

        % 条件C: 攻撃検出処理
        if cond == "C" && epoch <= detect_epochs
            cos_sims = zeros(K,K);
            for i = 1:K
                for j = 1:K
                    if i ~= j
                        g_i = grads(:,i);
                        g_j = grads(:,j);
                        if norm(g_i) ~= 0 && norm(g_j) ~= 0
                            cos_sims(i,j) = (g_i' * g_j) / (norm(g_i) * norm(g_j));
                        end
                    end
                end
            end
            for i = 1:K
                sum_cos = sum(cos_sims(i,:));
                if sum_cos <= tau
                    suspect_counts(i) = suspect_counts(i) + 1;
                end
            end
            if epoch == detect_epochs
                detected_attackers = find(suspect_counts >= acc_threshold_ratio * detect_epochs);
                fprintf("条件C 検出された攻撃者: %s\n", mat2str(detected_attackers));
            end
        end

        % 勾配集約・モデル更新
        global_grad = mean(grads, 2);
        w_global = w_global - mu * global_grad;

        % 記録
        acc_history(epoch) = mean(acc_clients);

        % ログ
        if mod(epoch,100) == 0 || epoch == epochs
            fprintf("条件%s Epoch %d: 精度 %.2f%%\n", cond, epoch, acc_history(epoch));
        end
    end

    results_acc{cond_idx} = acc_history;
end

% =========================
% グラフ描画
% =========================
figure; hold on;
colors = {'b','r','g'};
for i = 1:length(conditions)
    plot(1:epochs, results_acc{i}, 'LineWidth', 1.5, 'Color', colors{i});
end
xlabel('Epoch'); ylabel('Accuracy (%)');
legend(conditions); title('精度推移'); grid on;
>>>>>>> 7d526a9 (変更)
