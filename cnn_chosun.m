clc,clear,close all
% CNN / chosun
% 지정 장소에서 촬영한 IR 이미지
% Revision date: 2021.10.19

% images : 28 x 28 x 60000
% labels : 1 x 60000
% mnist\testingData.mat
% images : 28 x 28 x 6000
% labels : 1 x 6000
% sk.boo
addpath CNN_function
%% set up
% training set (MNIST)
load mnist_dataset\trainingData.mat;
% 학습데이터 간소화
% images = images(:,:,1:6000);
% labels = labels(1:6000);
x = reshape(images, 28,28,1,[]);
one_hot = diag(ones(1,max(labels)+1));
y = one_hot(labels+1,:)';

% test set (MNIST)
load mnist_dataset\testingData.mat;

% 커널 크기
kernelSize1 = [3 3 1 10];
kernelSize2 = [4 4 10 10];

%풀링 사이즈
poolDim1 = 2;
poolDim2 = 2;

%모델
imageDim = size(x,1);
labelClasses = size(y,1);
layerDim1 = ( imageDim - kernelSize1(1) + 1 )/poolDim1;
layerDim2 = ( layerDim1 - kernelSize2(1) + 1 )/poolDim2;
layerDim3 = layerDim2^2*kernelSize2(4);
kernelSize3 = [labelClasses layerDim3];

% 학습률
lr = 0.01;
%weight decay
lambda = 0.0001;
alpha_m = 0.5;
alpha_r = 0.99;

%세대 수 (epoch)
epo = 3;

%배치 수
batch = 150;

% 커널
U1 = (1e-1)*randn(kernelSize1(1),kernelSize1(2),kernelSize1(3),kernelSize1(4)); % a*b
U2 = (1e-1)*randn(kernelSize2(1),kernelSize2(2),kernelSize2(3),kernelSize2(4)); % c*d
U3 = (1e-1)*randn(kernelSize3(1),kernelSize3(2));  % r*q
B1 = zeros(kernelSize1(4), 1);
B2 = zeros(kernelSize2(4), 1);
B3 = zeros(labelClasses, 1);
% momentum
v1 = zeros(size(U1));% a*b
v2 = zeros(size(U2));% c*d
v3 = zeros(size(U3));% r*q
vb1 = zeros(size(B1));
vb2 = zeros(size(B2));
vb3 = zeros(size(B3));
% adaptive learnig rate
r1 = zeros(size(U1));% a*b
r2 = zeros(size(U2));% c*d
r3 = zeros(size(U3));% r*q
rb1 = zeros(size(B1));
rb2 = zeros(size(B2));
rb3 = zeros(size(B3));

idx=0;
tic;
%%
for e = 1:epo
    % data shuffle
    p = randperm(length(x));
    
    for i = 1:batch:(length(x)-batch+1)
        %Correlation(in,kernel,stride,padding)
        %Pooling(in,num,stride)
        X = x(:,:,:,p(i:i+batch-1)); %i*j
        Y = y(:,p(i:i+batch-1));
        X_num = size(X,4);
        idx=idx+1;
        
        % momentum
        if i == 20*batch+1
            alpha_m = 0.9;
        end
        
        dU1 = zeros(size(U1));% a*b
        dU2 = zeros(size(U2));% c*d
        dU3 = zeros(size(U3));% r*q
        dB1 = zeros(size(B1));
        dB2 = zeros(size(B2));
        dB3 = zeros(size(B3));
        
        % 입력데이터 정규화
        %         X =  Relu(Normalization3(X));
        X = X/255;
        
        z1 = Correlation(X,U1,B1);
        %         z1 = Normalization3(z1);
        layer1 = Relu(z1);% m*n
        [pool_layer1,pool_grad1] = Pooling(layer1,poolDim1,'mean'); % m'*n'
        
        z2 = Correlation(pool_layer1,U2,B2);
        %         z2 = Normalization3(z2);
        layer2 = Relu(z2);% o*p
        [pool_layer2,pool_grad2] = Pooling(layer2,poolDim2,'mean'); % o'*p'
        
        flat_layer3 = reshape(pool_layer2,[],X_num); % q*1
        out_layer = U3*flat_layer3 + B3; % r*1
        %         out_layer = (out_layer-mean(out_layer))./std(out_layer);
        
        % softmax error
        out = exp(out_layer)./sum(exp(out_layer),1);
        wCost = lambda/2 * (sum(U3(:).^2)+sum(U2(:).^2)+sum(U1(:).^2));
        error(idx,:) = -sum(sum(Y.*log(out)))/batch + wCost;
        %% Backpropagation
        gradient3 = out - Y; %out error gradient
        
        % o*p / CONV - FOOL - FC
        gradient_FC2 = reshape(U3' * gradient3,layerDim2,layerDim2,kernelSize2(4),X_num);
        gradient2 = UpSampling(gradient_FC2,poolDim2,'mean',pool_grad2).*ReluGradient(z2);
        
        % m*n / CONV - FOOL - CONV
        gradient_CONV1 = Convolution(gradient2,U2);
        gradient1 = UpSampling(gradient_CONV1,poolDim1,'mean',pool_grad1).*ReluGradient(z1);
        
        dU3 = gradient3*flat_layer3';
        dB3 = sum(gradient3,2);
        [dU2,dB2] = Update_grad(dU2,dB2,pool_layer1,gradient2);
        [dU1,dB1] = Update_grad(dU1,dB1,X,gradient1);
        
        %% Momentum
        v3 = alpha_m*v3 + (1-alpha_m)*dU3/batch;    hv3 = v3/(1-(alpha_m)^idx);
        vb3 = alpha_m*vb3 + (1-alpha_m)*dB3/batch;  hvb3 = vb3/(1-(alpha_m)^idx);
        v2 = alpha_m*v2 + (1-alpha_m)*dU2/batch;    hv2 = v2/(1-(alpha_m)^idx);
        vb2 = alpha_m*vb2 + (1-alpha_m)*dB2/batch;  hvb2 = vb2/(1-(alpha_m)^idx);
        v1 = alpha_m*v1 + (1-alpha_m)*dU1/batch;    hv1 = v1/(1-(alpha_m)^idx);
        vb1 = alpha_m*vb1 + (1-alpha_m)*dB1/batch;  hvb1 = vb1/(1-(alpha_m)^idx);
        
        %% RMSProp
        r3 =  alpha_r*r3 + (1-alpha_r)*(dU3/batch).^2;  hr3 = r3/(1-(alpha_r)^idx);
        rb3 =  alpha_r*rb3 + (1-alpha_r)*(dB3/batch).^2;hrb3 = rb3/(1-(alpha_r)^idx);
        r2 =  alpha_r*r2 + (1-alpha_r)*(dU2/batch).^2;  hr2 = r2/(1-(alpha_r)^idx);
        rb2 =  alpha_r*rb2 + (1-alpha_r)*(dB2/batch).^2;hrb2 = rb2/(1-(alpha_r)^idx);
        r1 =  alpha_r*r1 + (1-alpha_r)*(dU1/batch).^2;  hr1 = r1/(1-(alpha_r)^idx);
        rb1 =  alpha_r*rb1 + (1-alpha_r)*(dB1/batch).^2;hrb1 = rb1/(1-(alpha_r)^idx);
        
        U3 = U3 - lr*(hv3./((1e-8) + sqrt(hr3)) + lambda*U3);
        B3 = B3 - lr*hvb3./((1e-8) + sqrt(hrb3));
        U2 = U2 - lr*(hv2./((1e-8) + sqrt(hr2)) + lambda*U2);
        B2 = B2 - lr*hvb2./((1e-8) + sqrt(hrb2));
        U1 = U1 - lr*(hv1./((1e-8) + sqrt(hr1)) + lambda*U1);
        B1 = B1 - lr*hvb1./((1e-8) + sqrt(hrb1));
        
        tex1 = mean(error);
        %         fprintf("오차: %0.4f  %2.4f \n",error(idx),i/length(x)*100)
%         fprintf("%2.0f epoch 진행도 : %2.4f %% 전체 학습 오차: %0.4f\n",e,i/length(x)*100,error(idx))
        if i <= 20*batch+1
            fprintf("%2.0f epoch 진행도 : %2.4f %% 전체 학습 오차: %0.4f\n",e,i/length(x)*100,error(idx))
        end
    end
    time = toc;
    
    testim = reshape(images, [28,28,1,10000]);
    testim = testim/255;
    
    z1 = Correlation(testim,U1,B1);
    %     z1 = Normalization3(z1);
    layer1 = Relu(z1);% m*n
    [pool_layer1,pool_grad1] = Pooling(layer1,poolDim1); % m'*n'
    
    z2 = Correlation(pool_layer1,U2,B2);
    %     z2 = Normalization3(z2);
    layer2 = Relu(z2);% o*p
    pool_layer2 = Pooling(layer2,poolDim2); % o'*p'
    
    flat_layer3 = reshape(pool_layer2,[],length(testim)); % q*1
    out_layer = U3*flat_layer3 + B3; % r*1
    %     out_layer = (out_layer-mean(out_layer))./std(out_layer);
    
    % softmax error
    out = exp(out_layer)./sum(exp(out_layer),1);
    [~,preds] = max(out,[],1);
    
    acc = sum((preds-1)==labels)/length(preds);
    fprintf('%2.0f epoch / Accuracy is %4.2f %%\n',e,acc*100);
    time
    figure(1)
    plot(e/length(error):e/length(error):e,error,'c');
    xlabel("Epoch");ylabel("Cost");hold on;
    plot(e/length(error):e/length(error):e,smoothdata(error),'b','LineWidth',1.5)
    hold off;
end





