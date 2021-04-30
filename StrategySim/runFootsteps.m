function [winner,gameplay_data] = runFootsteps(P1_strategy,P2_strategy,GameState,verbose)

% waterFilling - train classifier against any combo of features with
%                   weights
% Syntax:  threshold = trainClassifier(trainList,featureDict,featVect,weigthVect)
%
% Inputs:
%    trainList - 
%    featureDict - 
%
% Outputs:
%    threshold - 
%------------- BEGIN CODE --------------

if nargin == 2 || isempty(GameState)
    GameState = [0 50 50];
end

if nargin < 4
    verbose = 0;
end


gameplay_data(1:2,:) = {'GameScore','S1','S2';GameState(1),GameState(2),GameState(3)};

t_step = 3;

while ~(GameState(1)==3 || GameState(1)==-3 || (GameState(2)+ GameState(3))==0)
    
    % Get Actions (Plays)
    if P1_strategy == 6
        action1 = agents(1,P1_strategy,GameState,P2_strategy);
    else
        action1 = agents(1,P1_strategy,GameState);
    end
    
    if P2_strategy == 6
        action2 = agents(2,P2_strategy,GameState,P1_strategy);
    else
        action2 = agents(2,P2_strategy,GameState);
    end
    
    % Update Player Scores
    GameState(2) = GameState(2) - action1;
    GameState(3) = GameState(3) - action2;
    
    % Update GameScore State
    if (action1 > action2)  
        if (GameState(1) <= 0)
            GameState(1) = GameState(1) - 1;
        else
            GameState(1) = -1;
        end
        
    elseif (action2 > action1)
        if (GameState(1) >= 0)
            GameState(1) = GameState(1) + 1;
        else
            GameState(1) = 1;
        end
        
    else
        GameState(1) = GameState(1);    
    end % if (action1 > action2)
    
    % Update GamePlay Data
    gameplay_data(t_step,:) = {GameState(1),GameState(2),GameState(3)};
    
    t_step = t_step + 1;

end % while ~done

winner = (GameState(1) > 0) + 1;
if verbose 
    fprintf(1,'GamePlayData: \n');
    disp(gameplay_data);
    fprintf(2,'\nWinner is Player %i...\n',winner);
end

end % function winner = runFootsteps(lambda,x)

