% =========================
% パラメータ設定
% =========================
K = 30;         % クライアント数
alpha = 1;      % ステップサイズ
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

% IID分割
N = size(X,1);
perm = randperm(N);
X = X(perm,:);
Y = Y(perm);
data_per_client = floor(N/K);
for k = 1:K
    idk = (k-1)*data_per_client + 1 : k*data_per_client;
    X_clients{k} = X(idk,:);
    Y_clients{k} = Y(idk);
end

% 攻撃者設定
attackers = randperm(K, floor(K * attacker_ratio));
fprintf("実際の攻撃者: %s\n", mat2str(attackers));
fprintf("正規クライアント: %s\n", mat2str(setdiff(1:K, attackers)));

% =========================
% 条件A/B/Cのループ
% =========================
conditions = {'A','B','C'};
results_acc = cell(1, length(conditions));
results_detect = cell(1, length(conditions));
results_detect_ratio = cell(1, length(conditions)); % 各エポックまでの割合

for cond_idx = 1:length(conditions)
    cond = conditions{cond_idx};
    fprintf("\n=== 条件%s 開始 ===\n", cond);

    % 重み初期化
    w_global = zeros(size(X,2),1);
    detected_attackers = [];
    suspect_counts = zeros(K,1); % 条件C用

    acc_history = zeros(1, epochs);
    detect_history = zeros(1, epochs);
    detect_ratio_history = zeros(1, epochs);

    for epoch = 1:epochs
        grads = zeros(size(X,2), K);
        acc_clients = zeros(1, K);

        % 各クライアント更新
        for k = 1:K
            Xk = X_clients{k};
            Yk = Y_clients{k};

            % 精度（条件A/B/C全てで計測）
            pred_labels = double((Xk * w_global) >= 0);
            acc_clients(k) = mean(pred_labels == Yk) * 100;

            % 勾配
            pred = 1 ./ (1 + exp(-Xk * w_global));
            grad = Xk' * (pred - Yk) / size(Xk,1);

            % 攻撃者の挙動
            if cond == "B" && ismember(k, attackers)
                grad = -grad;
            elseif cond == "C" && ismember(k, attackers) && epoch <= detect_epochs
                grad = -grad;
            end

            grads(:,k) = grad;
        end

        % 条件C: 攻撃検出処理
        if cond == "C"
            if epoch <= detect_epochs
                % コサイン類似度
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

                % τ以下のクライアントをカウント
                for i = 1:K
                    sum_cos = sum(cos_sims(i,:));
                    if sum_cos <= tau
                        suspect_counts(i) = suspect_counts(i) + 1;
                    end
                end

                % detect_epochsで検出確定
                if epoch == detect_epochs
                    detected_attackers = find(suspect_counts >= acc_threshold_ratio * detect_epochs);
                    fprintf("条件C 検出された攻撃者: %s\n", mat2str(detected_attackers));
                end
            end
        end

        % 勾配集約
        if cond == "C" && epoch > detect_epochs && ~isempty(detected_attackers)
            valid_clients = setdiff(1:K, detected_attackers);
            global_grad = mean(grads(:, valid_clients), 2);
        else
            global_grad = mean(grads, 2);
        end

        % モデル更新
        w_global = w_global - mu * global_grad;

        % 記録
        acc_history(epoch) = mean(acc_clients);
        detect_history(epoch) = length(detected_attackers);
        detect_ratio_history(epoch) = length(intersect(detected_attackers, attackers)) / length(attackers);

        % ログ
        if mod(epoch,100) == 0 || epoch == epochs
            fprintf("条件%s Epoch %d: 精度 %.2f%%, 検出割合 %.2f\n", ...
                cond, epoch, acc_history(epoch), detect_ratio_history(epoch));
        end
    end

    results_acc{cond_idx} = acc_history;
    results_detect{cond_idx} = detect_history;
    results_detect_ratio{cond_idx} = detect_ratio_history;
end

% =========================
% グラフ描画
% =========================
figure;
subplot(3,1,1);
hold on;
for i = 1:length(conditions)
    plot(1:epochs, results_acc{i}, 'LineWidth', 1.5);
end
xlabel('Epoch'); ylabel('Accuracy (%)');
legend(conditions); title('精度推移');
grid on;

subplot(3,1,2);
hold on;
for i = 1:length(conditions)
    plot(1:epochs, results_detect{i}, 'LineWidth', 1.5);
end
xlabel('Epoch'); ylabel('検出数');
legend(conditions); title('検出数推移');
grid on;

subplot(3,1,3);
hold on;
for i = 1:length(conditions)
    plot(1:epochs, results_detect_ratio{i}, 'LineWidth', 1.5);
end
xlabel('Epoch'); ylabel('検出割合');
legend(conditions); title('攻撃者検出割合');
grid on;
