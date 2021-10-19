function y = Normalization3(x)
arguments
    x (:,:,:,:) double
end
    y = (x-mean(x,[1,2]))./std(x,0,[1,2]);
end
