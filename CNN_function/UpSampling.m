% 21.06.29 sk.boo
% 풀링레이어의 미분
function out = UpSampling(in,pool_size,method,pool_grad)
arguments
    in (:,:,:,:) double % 풀링레이어 전의 행렬
    pool_size (1,1) double % 풀링사이즈
    method = "mean"
    pool_grad (:,:,:,:) double = 0

end

[col,row,ch,num] = size(in);
out = zeros(col*pool_size,row*pool_size,ch,num);
if method == "max"
    for n = 1:num
        for c = 1:ch
            unpool = in(:,:,c,n);
            out(:,:,c,n) = kron(unpool,ones(pool_size)).*pool_grad(:,:,c,n);
        end
    end
elseif method == "mean"
    for n = 1:num
        for c = 1:ch
            unpool = in(:,:,c,n);
            out(:,:,c,n) = kron(unpool,ones(pool_size))./(pool_size ^ 2);
        end
    end
else
    fprintf("method는 max(defult)또는 mean을 입력하세요")
    
end