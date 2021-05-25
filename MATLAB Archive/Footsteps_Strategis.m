% G = 0
figure;

subplot(1,3,1);
histogram(gamrnd(2,4,1,10000),'Normalization','pdf')

xlim([0 20]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Long Approach ~ Gamma(2,4)','FontWeight','bold','FontSize',12);

subplot(1,3,2);
histogram(20-gamrnd(2,4,1,10000),'Normalization','pdf')

xlim([0 20]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Short Approach ~ 20-Gamma(2,4)','FontWeight','bold','FontSize',12);

subplot(1,3,3);
histogram(normrnd(10,1,1,10000),'Normalization','pdf')

xlim([0 20]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Average Approach ~ Norm(10,1)','FontWeight','bold','FontSize',12);

sgtitle('First Move Strategies (G=0)','FontWeight','bold','FontSize',15);

% G = -1, Winning 1
S_self = 30;
S_opponent = 20;
figure;

subplot(1,3,1);
histogram(normrnd(S_opponent/2,1,1,10000),'Normalization','pdf')

xlim([0 30]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Long Approach ~ Norm(S2*0.5,1)','FontWeight','bold','FontSize',12);

subplot(1,3,2);
histogram(S_self/2 - gamrnd(2,4,1,10000),'Normalization','pdf')

xlim([0 30]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Short Approach ~ P1*0.5-Gamma(2,4)','FontWeight','bold','FontSize',12);

subplot(1,3,3);
histogram(normrnd(S_opponent/2,1,1,10000),'Normalization','pdf')

xlim([0 30]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Average Approach ~ Norm(S2*0.5,1)','FontWeight','bold','FontSize',12);

sgtitle({'Up 1 Move Strategies', 'Self: 30 Opp: 20'},'FontWeight','bold','FontSize',15);

% G = -2, Winnning 2
S_self = 30;
S_opponent = 20;

figure;
subplot(1,3,1);
histogram(S_opponent + 1 - gamrnd(1,2,1,10000),'Normalization','pdf')

xlim([0 31]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Long Approach ~ S2 + 1 - Gamma(1,2)','FontWeight','bold','FontSize',12);

subplot(1,3,2);
histogram(normrnd(S_self,.00001,1,10000),'Normalization','pdf')

xlim([0 31]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Short Approach ~ S1','FontWeight','bold','FontSize',12);

subplot(1,3,3);
histogram(normrnd(S_opponent/2,1,1,10000),'Normalization','pdf')

xlim([0 31]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Average Approach ~ Norm(S2*0.5,1)','FontWeight','bold','FontSize',12);

sgtitle({'Up 2 (Closing) Move Strategies', 'Self: 30 Opp: 20'},'FontWeight','bold','FontSize',15);

% G = 2, Losing 1
S_self = 30;
S_opponent = 20;

figure;

subplot(1,3,1);
histogram(gamrnd(2,4,1,10000),'Normalization','pdf')

xlim([0 31]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Long Approach ~ Gamma(2,4)','FontWeight','bold','FontSize',12);

subplot(1,3,2);
histogram(normrnd(S_opponent/2,1,1,10000),'Normalization','pdf')

xlim([0 31]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Short Approach ~ Norm(S2*0.5,1)','FontWeight','bold','FontSize',12);

subplot(1,3,3);
histogram(normrnd(S_opponent/2,1,1,10000),'Normalization','pdf')

xlim([0 31]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Average Approach ~ Norm(S2*0.5,1)','FontWeight','bold','FontSize',12);

sgtitle({'Down 1 Move Strategies', 'Self: 30 Opp: 20'},'FontWeight','bold','FontSize',15);

% G = 2, Losing 2
S_self = 30;
S_opponent = 20;

figure;

subplot(1,3,1);
histogram(S_opponent + 1 - gamrnd(1,2,1,10000),'Normalization','pdf')

xlim([0 31]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Long Approach ~ P2 + 1 - Gamma(1,2)','FontWeight','bold','FontSize',12);

subplot(1,3,2);
histogram(S_opponent + 1 - gamrnd(1,2,1,10000),'Normalization','pdf')

xlim([0 31]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Short Approach ~ P2 + 1 - Gamma(1,2)','FontWeight','bold','FontSize',12);

subplot(1,3,3);
histogram(normrnd(S_opponent/2,1,1,10000),'Normalization','pdf')

xlim([0 31]);
xlabel('Points','FontWeight','bold','FontSize',11);
ylabel('Prob','FontWeight','bold','FontSize',11);
title('Average Approach ~ Norm(S2*0.5,1)','FontWeight','bold','FontSize',12);

sgtitle({'Down 2 (One the Ropes) Move Strategies', 'Self: 30 Opp: 20'},'FontWeight','bold','FontSize',15);



