function [lambda, gamma] =  RecKT_CV(E_All, F_All, X_All, Y_All, list_All, lambda_candidate, gamma_candidate, hasLocality)
    K = 5;
    N = size(E_All,2);
    dim_f = size(F_All,1);
    N_u = floor(N/K);
    N_s = N - N_u;
    lambda_best = zeros(K,1);
    gamma_best = zeros(K,1);
    acc_matrix = zeros(length(lambda_candidate), length(gamma_candidate));

    if length(lambda_candidate) == 1 && length(gamma_candidate) == 1
       fprintf('Only one parameter, CV is not implemented!\n');
       lambda = lambda_candidate;
       gamma = gamma_candidate;
       return;
    end
    for k = 1:K
        fprintf('CV k = %d\n',k);
        fprintf('--------------------------------------------------------------------------------------------------\n');
        fprintf('--------------------------------------------------------------------------------------------------\n');
        t0 = clock();
        
        E_All_R = [E_All(:,[1:(k-1)*N_u,k*N_u+1:end]), E_All(:,(k-1)*N_u+1:k*N_u)];
        F_All_R = [F_All(:,[1:(k-1)*N_u,k*N_u+1:end]), rand(dim_f,N_u)];
        index = [];
        for i = 1:N_u
            index = [index;find(Y_All==list_All((k-1)*N_u+i))];
        end
                
        bestResult = 0;
        ind_i = 0; 
        for lambda = lambda_candidate
            ind_i = ind_i + 1;
            ind_j = 0;
            for  gamma = gamma_candidate
                ind_j = ind_j + 1;
                [alpha, F_u] = RecKT(E_All_R, F_All_R, N_s, lambda, gamma, hasLocality);
                [accuracy_Rec,Labels_predict] = classifier_nearest(X_All(index,:),F_u',list_All((k-1)*N_u+1:k*N_u),Y_All(index,:),1);
                acc_matrix(ind_i,ind_j) = acc_matrix(ind_i,ind_j) + accuracy_Rec;
            end
        end
        
        t1 = clock();
        fprintf('each Cross-Validation time cost = %f\n',etime(t1,t0));
    end
    
    [i,j] = ind2sub(size(acc_matrix),find(acc_matrix==max(max(acc_matrix))));
    lambda = lambda_candidate(i(1));
    gamma = gamma_candidate(j(1));

end