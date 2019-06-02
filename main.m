clc;clear;

acc = zeros(5,3,4);
isSmallSmaple = 0;
dataset = 'AwA'

for iter = 1:10
    fprintf('iter = %d\n', iter);

    %% AwA
    if dataset == 'AwA';
        load('AwA_ImageFeatures_VGG.mat')
        load('AwA_WordVectors.mat');
        load('AwA_Attributes.mat');
        Attribute = attributes_embedding_c;
        [~,dim_f] = size(ImageFeatures);
        load('AwA_splits_default.mat');
        load('AwA_ClassName.mat');
        classes_en = classes;
        N = 50;
        N_s = 40;
        N_u = 10;
        hasAttribute = 1;
        lambda_candidate = power(10,[-2:0.2:2]);
        gamma_candidate = power(10,[-2:0.2:0]);
        hasTop5 = 0;
        SparsitySet = [1];  % with/without sparsity
        LocalitySet = [0]; % with/without D

    %% CUB
    elseif dataset = 'CUB';
         load('CUB_ImageFeatures_ResGoog.mat');
         load('CUB_Attributes.mat');
         load('CUB_WordVectors.mat');
         WordVectors = K;
         [~,dim_f] = size(ImageFeatures);
         load('CUB_splits_default.mat');
         load('CUB_ClassName.mat');
         classes_cn = CUB_classes_cn(:,2);
         N = 200;
         N_s = 150;
         N_u = 50;
         hasAttribute = 1;
         lambda_candidate = power(10,[-2:0.5:0]);
         gamma_candidate = power(10,[-2:0.5:0]);
         hasTop5 = 0;
         SparsitySet = [1];  % with/without sparsity
         LocalitySet = [0]; % with/without D

    %% ImageNet
    elseif dataset = 'ImageNet';
         load('ImageNet_ImageFeatures_VGG_WordVectors.mat');
         [~,dim_f] = size(ImageFeatures);
         [~,dim_w] = size(WordVectors);
         load('ImageNet_splits_default.mat');
         load('ImageNet_ClassName.mat');
         classes_cn = ImageNet_classes_cn(:,2);
         N = 1360;
         N_s = 1000;
         N_u = 360;
         % errors
         errid = [545, 1353, 1309, 1131]; % these word vectors have some problem (conflict with others), here I randomly change their values
         for i = errid
             for j = 30:15:482
                 WordVectors(i,j) = WordVectors(i,dim_w-j);
             end
         end
         Attribute = WordVectors;
         hasAttribute = 0;
         lambda_candidate = power(10,[-1:0.2:0]);
         gamma_candidate =  [0];
         hasTop5 = 1;
         SparsitySet = [1];  % with/without sparsity
         LocalitySet = [0,1]; % with/without D

    %% Normalization
    ImageFeatures = ImageFeatures./max(max(ImageFeatures));
    Attribute = Attribute./max(max(Attribute));
    WordVectors = WordVectors./max(max(WordVectors));

    
    
    disp('***********************************************************************************************');

    list_all = [1:N]';
    list_test = splits';
    list_train = list_all;
    list_train(list_test) = [];
    list_train = list_train(randperm(numel(list_train))); % shuffle training set.
    disp(list_test');

    tn = 0;
    ind_i = 0;
    
        for  hasSparsity = SparsitySet
            if hasSparsity == 0
                lambda_candidate = 0;
                LocalitySet = 0;
            end
            
            for hasLocality = LocalitySet

                ind_i = ind_i + 1;
                fprintf('hasSparsity = %d, hasLocality = %d\n', hasSparsity, hasLocality);

                FeaTrain = []; % mean vectors of image features in each seen class
                WorTrain = []; % word vectors of each seen class name
                AttTrain = [];
                X_Train = [];
                Y_Train = [];
                for i = 1:length(list_train)
                    index = find(Labels==list_train(i));
                    x = ImageFeatures(index,:);
                    y = Labels(index,:);
                    X_Train = [X_Train;x];
                    Y_Train = [Y_Train;y];
                    FeaTrain = [FeaTrain;mean(x,1)];
                    WorTrain = [WorTrain;WordVectors(list_train(i),:)];
                    AttTrain = [AttTrain;Attribute(list_train(i),:)];
                end
                
                FeaTest = [];
                WorTest = [];
                AttTest = [];
                X = [];
                Y = [];
                for i = 1:length(list_test)
                    index = find(Labels==list_test(i));
                    X = [X;ImageFeatures(index,:)];
                    Y = [Y;Labels(index)];
                    FeaTest = [FeaTest;mean(ImageFeatures(index,:),1)];
                    WorTest = [WorTest;WordVectors(list_test(i),:)];
                    AttTest = [AttTest;Attribute(list_test(i),:)];
                end
                
                WorAll = [WorTrain;WorTest];
                AttAll = [AttTrain;AttTest];
                ConTrain = [AttTrain,WorTrain];
                ConTest = [AttTest,WorTest];
                ConAll = [ConTrain;ConTest];

                %% SSC
                % Cross-Validation
                % Attribute
                if hasAttribute
                    semantics = 'Attritbue';
                    E_s = AttTrain';
                    F_s = FeaTrain';
                    X_s = X_Train;
                    Y_s = Y_Train;
                    list_s = list_train;
                    [lambda, gamma] = RecKT_CV(E_s, F_s, X_s, Y_s, list_s, lambda_candidate, gamma_candidate, hasLocality);
                    
                    FeaRecon = rand(N_u,dim_f);
                    F_All = [FeaTrain;FeaRecon]';
                    E_All = [AttTrain;AttTest]';
                    [alpha_A, F_u_A, visresult] = RecKT(E_All, F_All, N_s, lambda, gamma, hasLocality);
                    [accuracy_A_Rec,Labels_predict] = classifier_nearest(X,(F_u_A'),list_test,Y,1);
                    fprintf('\n---------------------------------------------------------\n');
                    fprintf('lambda = %f, gamma = %f, accuracy_A_Rec = %f\n\n',lambda, gamma, accuracy_A_Rec);
                    acc(iter,ind_i,1) = accuracy_A_Rec;
                end
                
                % WordVector
                semantics = 'WordVector';
                E_s = WorTrain';
                F_s = FeaTrain';
                X_s = X_Train;
                Y_s = Y_Train;
                list_s = list_train;
                [lambda, gamma] = RecKT_CV(E_s, F_s, X_s, Y_s, list_s, lambda_candidate, gamma_candidate, hasLocality);
                
                FeaRecon = rand(N_u,dim_f);
                F_All = [FeaTrain;FeaRecon]';
                E_All = [WorTrain;WorTest]';
                [alpha_W, F_u_W, visresult] = RecKT(E_All, F_All, N_s, lambda, gamma, hasLocality);
                [accuracy_W_Rec,Labels_predict] = classifier_nearest(X,(F_u_W'),list_test,Y,1);
                fprintf('\n---------------------------------------------------------\n');
                fprintf('lambda = %f, gamma = %f, accuracy_W_Rec = %f\n\n',lambda, gamma, accuracy_W_Rec);
                acc(iter,ind_i,2) = accuracy_W_Rec;
                
                % A + W
                if hasAttribute
                    [accuracy_AW_Rec,Labels_predict] = classifier_nearest(X,(F_u_A'+F_u_W')./2,list_test,Y,1);
                    fprintf('\n---------------------------------------------------------\n');
                    fprintf('lambda = %f, gamma = %f, accuracy_AW_Rec_1 = %f\n\n',lambda, gamma, accuracy_AW_Rec);
                    acc(iter,ind_i,3) = accuracy_AW_Rec;
                end
                
                % Top 5 accuracy for ImageNet
                if hasTop5
                    [accuracy_Top5,Labels_predict] = classifier_nearest(X,(F_u_W'),list_test,Y,5);
                    fprintf('\n---------------------------------------------------------\n');
                    fprintf('lambda = %f, gamma = %f, Top5 accuracy_W_Rec = %f\n\n',lambda, gamma, accuracy_Top5);
                    acc(iter,ind_i,4) = accuracy_Top5;
                end
            end
        end
end

acc_mean = zeros(3,4);
for i = 1:3
    for j = 1:4
        acc_mean(i,j) = mean(acc(1:iter,i,j));
    end
end





%% clustering

alpha = alpha_A;
Clt_W = (abs(alpha)+abs(alpha'))./2;
Clt_D = diag(sum(Clt_W,2));
Clt_L =  (diag(diag(Clt_D).^(-0.5)))*(Clt_D - Clt_W)*(diag(diag(Clt_D).^(-0.5)));
[Clt_eV,Clt_eD] = eig(Clt_L);
%     Clt_eD = abs(Clt_eD);
Clt_eD = diag(Clt_eD);
[~,index] = sort(Clt_eD,'ascend');
Clt_eD = Clt_eD(index);
Clt_eD = diag(Clt_eD);
Clt_eV = Clt_eV(:,index);

Clt_T = 10;floor(sqrt(N)*4); % top eig values
Clt_Y = Clt_eV(:,1:Clt_T);


K = 11;
idx = kmeans(Clt_Y,K);
names = [classes_cn(list_train);classes_cn(list_test)];
clusters = cell(20,20);
for i = 1:K
     pos = find(idx==i);
     for j = 1:length(pos)
         fprintf('%s,    ',names{pos(j)});
         clusters(i,j) = {names{pos(j)}};
     end
     fprintf('\n'); fprintf('-----------------------------------------\n');
end