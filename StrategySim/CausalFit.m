load('Rand_Agent_Data_Big.mat')

avgData = squeeze(mean(GameplayData,1));
X = zeros(12750,3);
y = zeros(12750,1);

i=1;

for iGame = 1:5
    for iSelf = 1:50
        for iOpponent = 1:51
            X(i,1:3) = [iGame,iSelf,iOpponent];
            y(i) = avgData(iGame,iSelf,iOpponent);
            i=i+1;
        end
    end
end

weights_Rand = X\y;
