%% __________________ NAIVE BAYES WITH MULTINOMIAL DOCUMENT _______________
% ____________________ and our spam dictionary ____________________________

%% PREPROCESSING (SAME AS DECISION TREES INITIAL PART)

data = readtable('spam.csv');
% convert table to dataset
dsM = table2dataset(data); 

% Emptying Columns for useless variables
dsM(:,'Var5') = [];
dsM(:,'Var4') = [];

% Change Variables Names
dsM.Properties.VarNames{1} = 'spam';
dsM.Properties.VarNames{2}= 'txt';
dsM.Properties.VarNames{3}= 'dummy';

% Convert variables to dummy 1 for ham and 2 for spam
dsM.dummy = grp2idx(dsM.spam);

%recode dummy variable: spam=1 ham=0 (previously spam=2 ham=1)
dsM.dummy(dsM.dummy== 1) = 0;
dsM.dummy(dsM.dummy== 2) = 1;



%% MULTINOMIAL DOCUMENT 

% CHANGE THE STRUCTURE OF ds database to fit the multinomial document
% structure: each of the columns created are vectors that count
%the frequency of the words that interest us. For instance,
%winner vector should be 0,1,0,2,0,1,0,0,0,0,3 and so on, not just
%0,1,0,1,0,1,0,0,0,0,1 like in the 'bernoulli' document before
                                     
% Winner Category
pat_win = ["win", "awarded", "won", "award", "prize", "claim", "congratulations", "selected", "gift"];
winner = count(dsM.txt,pat_win);

% Attention-related category
pat_attention = ["urgent", "1st", "stop", "latest", "cash"]; % Making an array of all the attention-related words.
attention = count(dsM.txt,pat_attention); 

% Mobile/Phone Related category:
pat_mobile = ["nokia", "txt", "phone", "reply", "chat", "line", "tone", "mobile", "mins", "camera", "msg"]; 
mobile = count(dsM.txt, pat_mobile); 

% Ads/offer Related words:
pat_ads = ["customer", "service", "services", "guaranted", "contact", "apply", "code", "entry", "unsubscribe"]; % 
ads = count(dsM.txt, pat_ads);


% Adult/Dating Related Words:
pat_adult = ["dating", "sexy", "sex", "xxx", "love"];
adult = count(dsM.txt, pat_adult);

%å character
strange_a= ['å'];
strange_a= count(dsM.txt, strange_a);

%Attaching Variable to the main dataset
dsM.winner = winner;
dsM.attention = attention;
dsM.mobile = mobile;
dsM.adult = adult;
dsM.ads = ads;
dsM.strange_a= strange_a; 

%deleting spam column
dsM.spam=[];
dsM.txt= [];


%% TRAIN/TEST division

Q = length(dsM);
Q = round(Q);
trainRatio = 0.8;
testRatio = 0.2;
[trainInd,testInd] = dividerand(Q,trainRatio,testRatio);

for i = 1:length(trainInd)
    v = trainInd(i);
    train_test(v,1)=1;
end

dsM.train = train_test;

%divide into test/train set
train_data = dsM(dsM.train==1,:);
test_data = dsM(dsM.train==0,:);

%delete train column 
dsM.train=[];
train_data.train=[];
test_data.train=[];

%Change class from dataset to matrix/vector
nbM = double(dsM);
train_databM = double(train_data); 
test_databM = double(test_data); 


%% VARIABLE PREPARATION 

% ham/spam vector
y1= train_databM(:,1);
y2= test_databM(:,1);

% categories of predictors (adult, winner...) 
x1= train_databM(:,2:end);
x2= test_databM(:,2:end);


%% RUN NAIVE BAYES WITH MULTINOMIAL DISTRIBUTION
MdlM = fitcnb(x1,y1,'Distribution','mvmn');

isGenRate = resubLoss(MdlM,'LossFun','ClassifErr')
%Accuracy is more or less 94%

%Now determine whether the algorithm generalizes.
oosGenRate = loss(MdlM,x2, y2)

%% OPTIMIZATION

Mdl_opt = fitcnb(x2, y2,'Distribution','mvmn','OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'))


%% ROC CURVE

%Naive bayes for test data
y2= logical(y2);
MdlM2 = fitcnb(x2,y2,'Distribution','mvmn');
respNB= double(y2);

%compute posterior probabilities
[~,score_nbM] = resubPredict(MdlM2);

%Compute the standard ROC curve using the scores from the naive Bayes classification.
[XnbM,YnbM,TnbM,AUCnbM] = perfcurve(respNB,score_nbM(:,MdlM2.ClassNames),1);

%% Plot roc curve
plot(XnbM,YnbM)
legend('Multinomial Naive Bayes','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');



