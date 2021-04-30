
tic;
n = 1000;


figure; grid on; hold on;

for j = 1:3
    winnerData = zeros(1,n);
    for i = 1:n
        [winnerData(i),~] = runFootsteps(6,5);
        percent = nnz(winnerData == 1)/i * 100;
        
%         if j == 1
%             plot(i,percent,'bx','Markersize',3);
%         elseif j==2
%             plot(i,percent,'rx','Markersize',3);
%         else
%             plot(i,percent,'gx','Markersize',3);
%         end

        disp(i);
        
    end
    
    percent = nnz(winnerData == 1)/n * 100;
    disp(['% Player 1 = ', num2str(percent)]);
    
    percentVec = (cumsum(winnerData == 1)./[1:n]) * 100;
    if j == 1
        plot(percentVec,'b');
    elseif j==2
        plot(percentVec,'r');
    else
        plot(percentVec,'g');
    end
    
    drawnow;
end

toc

xlabel('Game Count','FontWeight','bold','FontSize',15);
ylabel('% Player 1 Wins','FontWeight','bold','FontSize',15);
title('Omni Mini-Max (P1) vs Naive (P2) Results','FontWeight','bold','FontSize',18);
legend('Run 1','Run 2','Run 3');
set(gca, 'XScale', 'log')