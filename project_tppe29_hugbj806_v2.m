%%%%%%%%%%%%%%%%%%%%%%%%%% Project TPPE29 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% Task 1, value a European call option without dividend%%%%%%%%%%
r = 0.0278;                     % Risk-free rate of return
sigma = 0.2;                    % Spread, Volatlity
t1 = datetime(2022,11,28);      % Start date
t2 = datetime(2023,7,21);       % End date
fiscal_periods = 8;             % Number of periods
T = days252bus(t1, t2);         % Duration
K = 55;                         % Strike price
S0 = 62;                        % Spot price (Underlying value)
type = "C-EU";                  % European Calloption
DIV= 0;                         % Dividend
up_and_out = 0;                 % Set to 1 means that we dont up_and_out
disp("Task 1")
CalcBinom(r, sigma, T, fiscal_periods, S0, K, type, 0, DIV, up_and_out);
%%%%%%%%%%% Task 1, value a European call option without dividend%%%%%%%%%%

%%%%% Task 2, value a European call (OMX30) option without dividend %%%%%%%
%a
t1 = datetime(2022,11,28);      % Start date
t2 = datetime(2023,3,17);       % End date
T = days252bus(t1, t2);         % Duration
K = 2100;                       % Strike price
S0 = 2096.462;                  % Spot price (Underlying value)
C = 95.50;                      % Ask
type = "C-EU";                  % European Calloption
up_and_out = 0;                 % Set to 1 means that we dont up_and_out

disp("Task 2 a")
FindSigma(S0, K, 1,  r, T/252, C);
disp(sigma2_a);

b_s = BlackScholesCall(S0, K, 0.1946, r, T/252);
%b
disp("Task 2 b")
disp(CalcBinom(r, sigma2_a, T, 10, S0, K, type, 0, DIV, up_and_out));

%c
disp("Task 2 c");
plot(OptionPrices(r, sigma2_a, T, 5, 200, S0, K, b_s));
disp("period " + close_period);

%%%%% Task 2, value a European call (OMX30) option without dividend %%%%%%%
%%%%%%%%% Task 3, value a Europeanor American calloption with dividend %%%%
r = 0.0278;  
t1 = datetime(2022,11,28);      % Start date
t2div = datetime(2023,3,21);    % End date dividend
t2 = datetime(2023,7,21);       % End date
T = days252bus(t1, t2);         % Duration
Tdiv = days252bus(t1, t2div);   % Duration of dividend
K = 55;                         % Strike price
S0 = 62;                        % Spot price (Underlying value)
fiscal_periods = 8;             % Number of periods
t_div = Tdiv/(252);             % Time to dividend (Y)
DIV = 8;                        % Dividend
sigma = 0.2;                    % Spread, Volatlity
type1 = "C-EU";                 % European Calloption
type2 = "C-AM";                 % American Calloption
up_and_out = 0;                 % Set to 0 means that we dont up-and-in

S_star = S0 - DIV*exp(-r*t_div);
disp("Task 3 ");
CalcBinom(r, sigma, T, fiscal_periods, S_star, K, type1, t_div, DIV, ...
    up_and_out)
CalcBinom(r, sigma, T, fiscal_periods, S_star, K, type2, t_div, DIV, ...
    up_and_out)


%%%%%% Task 3, value a European or American calloption with dividend %%%%%%%

%%%%%%%%%%% Task 4, value American calloption with dividend %%%%%%%%%%%%%%%
r = 0.0278;  
t1 = datetime(2022,11,28);      % Start date
t2div = datetime(2023,4,13);    % End date dividend
t2 = datetime(2023,9,15);       % End date
T = days252bus(t1, t2);         % Duration
Tdiv = days252bus(t1, t2div);   % Duration of dividend
K = 430;                        % Strike price
S0 = 385.90;                    % Spot price (Underlying value)
fiscal_periods = 200;           % Number of periods
t_div = Tdiv/(252);             % Time to dividend
DIV = 4.9;                      % Dividend
sigma = 0.3593;                 % Spread, Volatlity
type = "C-AM";                  % American Calloption
up_and_out = 0;                 % Set to 1 means that we dont up-and-out

S_star = S0 - DIV*exp(-r*t_div);
disp("Task 4");
[~, binom] = CalcBinom(r, sigma, T, fiscal_periods, S_star, K, type, t_div, ...
    DIV, up_and_out);
disp(binom);
%%%%%%%%%%% Task 4, value a American calloption with dividend %%%%%%%%%%%%%

%%%%%%%%%%% Task 5, value a American calloption with dividend %%%%%%%%%%%%%

%%%%%%%%%%% Task 5, value a American calloption with dividend %%%%%%%%%%%%%
t1 = datetime(2022,11,28);      % Start date
t2 = datetime(2023,3,17);       % End date
T = days252bus(t1, t2);         % Duration
K = 2100;                       % Strike price
S0 = 2096.462;                  % Spot price (Underlying value)
C = 95.50;
up_and_out = 1;                  % Set to 1 means that we use up_and_out
B = S0*1.05;                    % Barrier level
type = "C-EU";
disp("Task 5")
[~, binom] = CalcBinom(r, sigma2_a, T, 30, S0, K, type, 0, DIV, up_and_out);
up_and_out_binom = binom;
disp("The up-and-out comes to (binomial): " + up_and_out_binom);
[~, binom] = CalcBinom(r, sigma2_a, T, 30, S0, K, type, 0, DIV, 0);
plain_vanilla_binom = binom;
disp("Our Plain Vanilla for binomial: " + plain_vanilla_binom);
up_and_in_binom= plain_vanilla_binom - up_and_out_binom;
disp("The up and in comes to(binomial mehtod): " + up_and_in_binom );
disp(" ");
disp("If we use the blackScholes method instead: ")
disp(" ");
up_and_out=UpAndOutFunc(S0,K,r,T,sigma2_a,B);
disp("The up-and-out comes to: " + up_and_out)
plain_vanilla_bs= BlackScholesCall(S0, K, sigma2_a, r, T/252);
up_and_in_bs = plain_vanilla_bs- up_and_out;
disp("The up and in comes to: " + up_and_in_bs );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function optionPrices = OptionPrices(r, sigma, T, start_period, end_period, ...
    S0, K, b_s)
temp=1;
    o_p = zeros(1,end_period -start_period);
    for x=1:end_period -start_period
        [~,binom]=CalcBinom(r, sigma, T, x, S0, K,"C-EU",0,0,0);
        o_p(x) = binom;
        if (o_p(x)< b_s + b_s*0.005) && (o_p(x) > (b_s - b_s*0.005) && temp)
            assignin('base','close_period',start_period + x)
            temp=0;
        end
    end
    optionPrices = o_p;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function newSigma = FindSigma(S0, K, sigma, r, T, C)
mid= sigma/2;
    newC = round(BlackScholesCall(S0, K, sigma, r, T), 2);
    
    newSigma = sigma;
    assignin('base','sigma2_a',newSigma)
        if newC < C
            sigma= (sigma + mid/2);
            FindSigma(S0, K,sigma, r, T, C);
        elseif newC > C
            sigma= mid/2;
            FindSigma(S0, K, sigma, r, T, C);
        end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [price] = UpAndOutFunc(S,K,r,T,sigma,H)

call_price = BlackScholesCall(S, K, sigma, r, T/252);
disp("Black & Scholes Price (Plain vanilla): " +call_price);

% Calc prob of K.O.
p_up = normcdf((log(H/S) + (r + sigma^2/2)*T)/(sigma*sqrt(T)));
disp("Probability of knock out: " + p_up);

% Calculate the option price
price = call_price * (1 - p_up);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function new_c = BlackScholesCall(S0, K, sigma, r, T)
d1 = ((log(S0/K)+ (r +((sigma^2)/2) )*T))/(sigma*sqrt(T));
d2 = d1 - sigma*sqrt(T);
new_c= S0 * normcdf(d1) - (K*exp(-r*T) * normcdf(d2));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [callOptionTree, binom] = CalcBinom(r, sigma, T, ...
    fiscal_periods, S0, K, type, t_div, DIV, up_and_out)
    delta_t = T/(fiscal_periods*252);
    u=exp(sigma*sqrt(delta_t));
    d=exp(-sigma*sqrt(delta_t));
    q=(exp(r*delta_t)-d)/(u-d);

    stockTree = zeros(fiscal_periods+1, fiscal_periods+1);
for col = 0:fiscal_periods
    for row =0:col
        stockTree(row + 1, col +1 ) = S0*d^row*u^(col-row);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EU %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if type == "C-EU"
    optionTree = zeros(fiscal_periods, fiscal_periods);
    for col =fiscal_periods +1 : -1:1
         for row =1:col
         if col==fiscal_periods +1
             optionTree(row , col ) = max(stockTree(row, col) - K, 0);
         else
                optionTree(row , col ) = exp(-r*delta_t) * (q*optionTree(row  , ...
                   col+1) + (1-q)*optionTree(row + 1 , col+1));
         end  
            if stockTree(row , col)>S0*1.05 && up_and_out
                optionTree(row, col)=0;
            else
            end
         end
    end

        callOptionTree=optionTree;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EU %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if type == "C-AM"
    optionTree = zeros(fiscal_periods, fiscal_periods);
    for col =fiscal_periods +1 : -1:1
         for row =1:col
            if col==fiscal_periods +1
             optionTree(row , col ) = max(stockTree(row, col) - K, 0);
            else
                 if floor(t_div/delta_t) == col -1
                optionTree(row , col ) =max(stockTree(row, col)+ DIV -K ...
                    ,exp(-r*delta_t) * (q*optionTree(row  , ...
                   col+1) + (1-q)*optionTree(row + 1 , col+1)));
                 else
                        optionTree(row , col ) = exp(-r*delta_t) * (q*optionTree(row  , ...
                        col+1) + (1-q)*optionTree(row + 1 , col+1));
                 end
            end
            % Hur kan man någonsin köra up and in om s0 ska va större än
            % s0*1.05

            if stockTree(row , col)>S0*1.05 && up_and_out
                optionTree(row, col)=0;
            else
            end

         end
    end
        callOptionTree=optionTree;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp(optionTree)
binom=optionTree(1,1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
