clc,clear,close all
load mnist_dataset\testingData.mat
% 학습 가중치
load cnn_adam_98.72.mat

% testData에서 6000개 중에서 선택
number = 3554;

one_hot = diag(ones(1,max(labels)+1));
y = one_hot(labels+1,:)';
Y = y(:,number);
testim = reshape(images, [28,28,1,10000]);
testim = testim(:,:,:,number);
testim = testim/255;
% 기준 zero 이미지
baseline = zeros(size(testim,[1 2 3]));

figure(1)
montage({baseline,testim});title('base image    input image');

grad_sum = zeros(26,26,10);
figure(2)
for i = 0:20
scaled_inputs = baseline + (i/20)*(testim-baseline);
%     imshow(scaled_inputs)

poolDim1 = 2; poolDim2 = 2;
%% forward
z1 = Correlation(scaled_inputs,U1,B1);
layer1 = Relu(z1);
pool_layer1 = Pooling(layer1,poolDim1);

z2 = Correlation(pool_layer1,U2,B2);
layer2 = Relu(z2);
pool_layer2 = Pooling(layer2,poolDim2);

flat_layer3 = reshape(pool_layer2,[],size(testim,4));
out_layer = U3*flat_layer3 + B3;

% softmax error
out = exp(out_layer)./sum(exp(out_layer),1);
% [~,preds] = max(out,[],1);
%% Backpropagation
gradient3 = out - Y; %out error gradient

% o*p / CONV - FOOL - FC
gradient_FC2 = reshape(U3' * gradient3,5,5,10);
gradient2 = UpSampling(gradient_FC2,poolDim2,'mean').*ReluGradient(z2);

% m*n / CONV - FOOL - CONV
gradient_CONV1 = Convolution(gradient2,U2);
gradient1 = UpSampling(gradient_CONV1,poolDim1,'mean').*ReluGradient(z1);
grad_sum = grad_sum+gradient1;
% figure(3)
% for j=1:10
% subplot(2,5,j)
% imshow(grad_sum(:,:,j))
% end
end
%%

fprintf(' 0 : %5.2f%% \n 1 : %5.2f%% \n 2 : %5.2f%% \n 3 : %5.2f%% \n 4 : %5.2f%% \n 5 : %5.2f%% \n 6 : %5.2f%% \n 7 : %5.2f%% \n 8 : %5.2f%% \n 9 : %5.2f%% \n',LogSigmoid(out_layer)*100);
sum_grad_sum = sum(grad_sum,3);
pad = zeros(size(testim));
pad(2:end-1,2:end-1) = sum_grad_sum;
pad_sum_grad_sum = pad;

a = (testim-baseline)*pad_sum_grad_sum;
load spine
figure(4)
montage({pad_sum_grad_sum,testim});