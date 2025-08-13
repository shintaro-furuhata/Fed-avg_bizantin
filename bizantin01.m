% =========================
% パラメータ設定
% =========================
K = 30;         % クライアント数
alpha = 1;      % ステップサイズ
mu = 0.01;      % 学習率
tau = -2;       % 提案手法の閾値
epochs = 1000;  % エポック数
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

% =========================
% 条件A/B/Cのループ
% =========================
conditions = {'A','B','C'};
results_acc = cell(1, length(conditions)); % 精度推移
results_detect = cell(1, length(conditions)); % 検出数推移

for cond_idx = 1:length(conditions)
    cond = conditions{cond_idx};
    fprintf("\n=== 条件%s 開始 ===\n", cond);
    
    % 重み初期化
    w_global = zeros(size(X,2),1);
    detected_attackers = [];
    
    acc_history = zeros(1, epochs);
    detect_history = zeros(1, epochs);
    
    for epoch = 1:epochs
        grads = zeros(size(X,2), K);
        acc_clients = zeros(1, K);
        
        % 各クライアント更新
        for k = 1:K
            Xk = X_clients{k};
            Yk = Y_clients{k};
            
            % 精度
            pred_labels = double((Xk * w_global) >= 0);
            acc_clients(k) = mean(pred_labels == Yk) * 100;
            
            % 勾配
            pred = 1 ./ (1 + exp(-Xk * w_global));
            grad = Xk' * (pred - Yk) / size(Xk,1);
            
            % 条件B/Cのみ攻撃者
            if ismember(k, attackers) && (cond ~= "A")
                if ~(cond == "C" && epoch >= 501 && ismember(k, detected_attackers))
                    grad = -grad; % sign-flipping
                end
            end
            
            grads(:,k) = grad;
        end
        
        % 条件C: 攻撃検出（501エポック以降）
        if cond == "C" && epoch >= 501
            % 98%以上の正答率 → 候補
            candidate_attackers = find(acc_clients >= 98);
            
            % コサイン類似度
            cos_sims = zeros(K,K);
            for k1 = 1:K
                for k2 = 1:K
                    if k1 ~= k2
                        g1 = grads(:,k1);
                        g2 = grads(:,k2);
                        if norm(g1) ~= 0 && norm(g2) ~= 0
                            cos_sims(k1,k2) = (g1' * g2) / (norm(g1) * norm(g2));
                        end
                    end
                end
            end
            mean_cos = sum(cos_sims, 2) / (K-1);
            detected_attackers = candidate_attackers(mean_cos(candidate_attackers) < tau);
        end
        
        % 勾配集約
        if cond == "C" && epoch >= 501 && ~isempty(detected_attackers)
            valid_clients = setdiff(1:K, detected_attackers);
            global_grad = mean(grads(:, valid_clients), 2);
        else
            global_grad = mean(grads, 2);
        end
        w_global = w_global - mu * global_grad;
        
        % 記録
        acc_history(epoch) = mean(acc_clients);
        detect_history(epoch) = length(detected_attackers);
        
        % ログ
        if mod(epoch,100) == 0 || epoch == epochs
            fprintf("条件%s Epoch %d: 精度 %.2f%%, 検出数 %d\n", ...
                cond, epoch, acc_history(epoch), detect_history(epoch));
        end
    end
    
    results_acc{cond_idx} = acc_history;
    results_detect{cond_idx} = detect_history;
end

% =========================
% グラフ描画
% =========================
figure;
subplot(2,1,1);
hold on;
for i = 1:length(conditions)
    plot(1:epochs, results_acc{i}, 'LineWidth', 1.5);
end
