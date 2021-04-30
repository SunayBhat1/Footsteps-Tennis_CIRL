% Collect Game Space Data

n=1000;
GameplayData = zeros(n,5,50,50);

for iN = 1:n
    for iGame = -2:2
        for iSelf = 1:50
            for iOpponent = 0:50

                GameplayData(iN,iGame+3,iSelf,iOpponent+1) = agents(1,4,[iGame iSelf iOpponent]);



            end
        end
    end
    disp(iN)
end

save("Rand_Agent_Data_Big.mat",'GameplayData')