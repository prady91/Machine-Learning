UBitName = 'pkortha';
personNumber = '50167960';
data=xlsread('data_new.xlsx');
%load('proj2.mat');
target = data(:,47);
features = data(:,1:46);
trainInd1 = (1:55698)';
validInd1 = (55699:62661)';
train = features(1:55698,:);
target_train = target(1:55698,:);
target_valid = target(55699:62661,:);
target_test = target(62662:69623,:);
%size(data)
format long g;
valid_data = features(validInd1,:);
valid_target = target(validInd1,:);
Mlimit = 8;
D=46;
M_fin = 2;
Erms_fin = 10.0;
lambda_fin = 0.0001;   
w_fin = ones(Mlimit,1);
mu_fin = zeros(Mlimit,D);
trainPer1 = 0;



for M=8:Mlimit
    
    lambda = 4;
    covMatrix = diag(var(train))*0.2+eye(D)*0.1;
    designMat_valid = ones(length(valid_data),M);
    designMat = ones(length(train),M);

    mu = ones(M,46); %MXD mean matrix
    for i =2:M
        ind = randperm(55698);
        muind = ind(1:50);
        tmp = mean(train(muind,:));
        while ismember(mu,tmp,'rows')
            fprintf('%s\n','enter');
            muind = ind(1:50);
            tmp = mean(train(muind,:));        
        end
        mu(i,:) = tmp;
        
        bas = zeros(length(train),1);
        for j = 1:length(train)
             x = train(j,:);
             x_mu = x-tmp;
            aux_row = (x_mu)/covMatrix*transpose(x_mu);
             aux_val = -1/2*aux_row; 
             bas(j,1) = exp(aux_val);  
        end
        designMat(:,i) = bas;
    
        bas = zeros(length(validInd1),1);
        for j = 1:length(validInd1)
            x = valid_data(j,:);
            x_mu = x-tmp;
            aux_row = (x_mu)/covMatrix*transpose(x_mu);
            aux_val = -1/2*aux_row;  
            bas(j,1) = exp(aux_val);
        end
        designMat_valid(:,i) = bas;
    end    
    
    while lambda<=4
        w_ml = ((inv(transpose(designMat)*designMat+lambda*eye(M)))*transpose(designMat))*(target_train);
        E_w = 0;
        if length(w_ml)>1
            w_er = w_ml(1:length(w_ml)-1,:);
            E_w = (transpose(w_er)*w_er)/2;
        end
        tmp = (target_train - (designMat*w_ml));
        E_d = (transpose(tmp)*tmp)*0.5;
    %    E = E_d+lambda*E_w;
        E = E_d;
        Erms_train = sqrt(2*E/length(train));
        
        
        tmp = (target_valid - designMat_valid*w_ml);
        E_d = (transpose(tmp)*tmp)/2;
        E = E_d;
        Erms_valid = sqrt(2*E/length(validInd1));
        
        if Erms_valid<Erms_fin
            Erms_fin = Erms_valid;
            M_fin = M;
            lambda_fin = lambda;
            mu_fin = mu;
            w_fin = w_ml;
            trainPer1 = Erms_train;
        end
        fprintf('%d %d %d %d\n',M,lambda,Erms_train,Erms_valid);
        lambda = lambda+0.1;
 %       w_ml 
        
    end
    
end
fprintf('%d \t%d \t%d \t%d \t%s\n',M_fin,lambda_fin,trainPer1,Erms_fin,'final');
M1 = M_fin;
mu1 = mu_fin';
lambda1 = lambda_fin;
w1 = w_fin;
validPer1 = Erms_fin;
Sigma1 = zeros(46,46,M1);
for i=1:M1
    Sigma1(:,:,i) = covMatrix;
end

save('proj2.mat');

