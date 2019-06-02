function [alpha, F_u, visresult] = RecKT(E_All, F_All, N_s, lambda, gamma, hasLocality)

visresult = cell(40,1); % for visualization and analysis
N = size(E_All,2);
t0 = clock();

% initialization
D = zeros(N,N); % each line represents the Di
alpha = rand(N,N);
for i = 1:N
    for j = 1:N
        if i~=j

            D(i,j) = log(1 + norm(E_All(:,i)-E_All(:,j),2));
        else
            D(i,j) = 0;
        end
    end
    D(i,:) = D(i,:)./mean(D(i,:));
    D(i,i) =1;
end

if hasLocality == 0
    D(:,:) = 1;% without D
end

alpha_old = alpha;
F_u_old = F_All(:,N_s+1:end);
iter_opt = 0;
V_one = ones(1,N);
flag_first = 1;
while 1
    iter_opt = iter_opt + 1;

    % Update alpha
    for i = 1:N
        e = E_All(:,i);
        E = E_All;
        E(:,i) = 0;
        f = F_All(:,i);
        F = F_All;
        F(:,i) = 0;
        rD = diag(D(i,:).^(-1)); %inv(Di)
        ED = E*rD;
        FD = F*rD;

        if flag_first==1||gamma==0
            beta = LeastR([ED], [e], lambda);
            flag_first = 0;
        else
            beta = LeastR([ED;gamma*FD], [e;gamma*f], lambda);
        end

        a= rD*beta;
        alpha(:,i) = a;
        clear a;
    end
    t3 = clock();

    if isnan(sum(sum(alpha)))
        fprintf('NaN error occurs in LeastR\n');
        F_u = F_u_old;
        break;
    end

    % Update F^u
    F_s = F_All(:,1:N_s);
    beta = eye(N,N) - alpha;
    beta_s = beta(1:N_s,:);
    beta_u = beta(N_s+1:end,:);
    F_u = -F_s*beta_s*pinv(beta_u);
    F_All(:,N_s+1:end) = F_u;

    visresult(iter_opt) = {F_u};
    isconverge = (sum(sum((alpha - alpha_old).^2))<0.0001 && sum(sum((F_u - F_u_old).^2))<0.0001) || iter_opt>30;

    alpha_old = alpha;
    F_u_old = F_u;

    if isconverge
        break;
    end

end

visresult = visresult(1:iter_opt);
t1 = clock();

end