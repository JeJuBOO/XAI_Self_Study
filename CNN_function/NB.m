%21.06.29 SK.boo
%같이 풀링되는 요소들 구하기
function y = NB(a,o,p,s)
    % a : 풀링전의 행렬
    % o,p : 풀링전의 행렬 인덱스
    % s : 풀링 사이즈
    % ex) a = 4*4size행렬에서 s = 2 (2*2size)풀링 에서 o,p = 3,2인덱스는
    %     a(3,1),a(3,2),a(4,1),a(4,2) 즉 a(3:4 , 1:2)의 요소를 추출
    y = a(o-mod(o-1,s):o-mod(o-1,s)+s-1 , p-mod(p-1,s):p-mod(p-1,s)+s-1,:,:);

end