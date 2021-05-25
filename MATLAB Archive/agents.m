function play = agents(id,strategy,GameState,strategy_opponent)
% agents - Outputs play based on game state (G,S1,S2) and agent id and staregytype
%
% Syntax:  threshold = trainClassifier(trainList,featureDict,featVect,weigthVect)
%
% Inputs:
%    trainList - 
%    featureDict - 
%
% Outputs:
%    threshold - 
%------------- BEGIN CODE --------------

if nargin == 3
    strategy_opponent = 0;
end

% Setup gamestate according to ID
if id == 1
    G = GameState(1);
    S_self = GameState(2);
    S_opponent = GameState(3);
else
    G = -GameState(1);
    S_self = GameState(3);
    S_opponent = GameState(2);
end

play = 0;
iter = 1;
% Repeat until play is within acceptable bounds 
while ~(play > 0 && play <= S_self)
    
    if iter == 10
%         fprintf('Playing Max Points');
        play = S_self;
        break;
    end
    
    if S_self == 0
        play = 0;
        break;
    end

    switch strategy
        % Mean Player
        case 1
            
            mean_play = S_opponent/2;
            if (G == 0)
                play = round(normrnd(10,1));   
            else
                play = round(normrnd(mean_play,1));
            end
            
        % Long Player
        case 2
            if (G == 0)
                play = round(gamrnd(2,4)); 
            elseif (G == 1) 
                play = round(gamrnd(2,4)); 
            elseif (G == 2)
                play = S_opponent + 1 - round(gamrnd(1,2));
            elseif (G == -1)
                mean_play = S_opponent/2;
                play = round(normrnd(mean_play,1));
            elseif (G == -2)
                play = S_opponent + 1 - round(gamrnd(1,2));
            end
            
        % Short Player
        case 3
            if (G == 0)
                play = 20 - round(gamrnd(2,4));
            elseif (G == 1)  
                mean_play = S_opponent/2;
                play = round(normrnd(mean_play,1));
            elseif (G == 2)
                play = S_opponent + 1 - round(gamrnd(1,2));
            elseif (G == -1)
                play = round(S_self/2) - round(gamrnd(2,4));
            elseif (G == -2)
                play = S_self;
            end
            
        % Uniform Random Player
        case 4

            play = round(rand()*S_self);
            
        % Naive MiniMax Player
        case 5
            % Best opponent startegy
            wins = zeros(4,1);
            for iStrategy = 1:4
                for nSample = 1:10
                    [winner,~] = runFootsteps(iStrategy,4,GameState);
                    wins(iStrategy) = wins(iStrategy) + (winner == 1);
                    
                end
                wins(iStrategy) = wins(iStrategy) /10;
            end
            
            [~,strategy_opponent] = max(wins);
            
            % Best response strategy
            wins = zeros(4,1);
            for iStrategy = 1:4
                for nSample = 1:10
                    [winner,~] = runFootsteps(iStrategy,strategy_opponent,GameState);
                    wins(iStrategy) = wins(iStrategy) + (winner == 1);
                    
                end
                wins(iStrategy) = wins(iStrategy) /10;
            end
            
            [~,strategy] = max(wins);
            
            play = agents(1,strategy,GameState);
            
        % Known MiniMax Player
        case 6
            % Best response strategy
            wins = zeros(4,1);
            for iStrategy = 1:4
                for nSample = 1:10
                    [winner,~] = runFootsteps(iStrategy,strategy_opponent,GameState);
                    wins(iStrategy) = wins(iStrategy) + (winner == 1);
                    
                end
                wins(iStrategy) = wins(iStrategy) /10;
            end
            
            [~,strategy] = max(wins);
            
            play = agents(1,strategy,GameState);
            
    end % switch type
    
    play = cast(play,'uint8');
    iter = iter + 1;
end % while play>0 && play <= P_self


end % function play = agents(type,G,P_self,P_opponent)