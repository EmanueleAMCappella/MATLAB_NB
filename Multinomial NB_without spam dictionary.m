

%% ______________ NAIVE BAYES WITH MULTINOMIAL DOCUMENT MODEL _____________
% __________________    but without our 'spam dictionary'  ________________

%% Data preprocessing

%open data
data = readtable('spam.csv');
% convert table to dataset
dsM2 = table2dataset(data); 

% Emptying Columns for useless variables
dsM2(:,'Var5') = [];
dsM2(:,'Var4') = [];

% Change Variables Names
dsM2.Properties.VarNames{1} = 'spam';
dsM2.Properties.VarNames{2}= 'txt';
dsM2.Properties.VarNames{3}= 'dummy';

% Convert variables to dummy 1 for ham and 2 for spam
dsM2.dummy = grp2idx(dsM2.spam);
%recode dummy variable: spam=1 ham=0 (previously spam=2 ham=1)
dsM2.dummy(dsM2.dummy== 1) = 0;
dsM2.dummy(dsM2.dummy== 2) = 1;


%% Create Bag of words

ds_text = dsM2.txt;     
% Erase Punctuation and get lowercase 
cleantxtTot = lower(erasePunctuation(ds_text));
% Remove stop words
dirttxt_test = tokenizedDocument(cleantxtTot); 
dirttxt_test= removeWords(dirttxt_test,stopWords); 
% Remove Long and short words   
dirttxt_test = removeShortWords(dirttxt_test,2);
dirttxt_test = removeLongWords(dirttxt_test,15);

%Create a bag-of-words model from the tokenized document.
bagTot = bagOfWords(dirttxt_test)
bagTot = removeInfrequentWords(bagTot,2);


%% Get corresponding matrix

totmatrix = bagTot.Counts;
totmatrix = full(totmatrix);
totmatrix = [totmatrix dsM2.dummy];
totmatrix = mat2dataset(totmatrix);


%% Train/test data generation

trainRatio = 0.8;
testRatio = 0.2;

%length of ds dataset
L = round(length(totmatrix));

%train/test division of ds database
[trainIndex,testIndex] = dividerand(L,trainRatio,testRatio);

for i = 1:length(trainIndex)
    pos = trainIndex(1,i);
    train_test(pos,1)=1;
end

totmatrix.train = train_test;
train_data = totmatrix(totmatrix.train==1,:);
test_data = totmatrix(totmatrix.train==0,:);

%create two vectors with dummy spam/ham labels
train_label = train_data(:,2579);
test_label = test_data(:,2579);

%delete from train and test data the column with train/test labels  
train_data(:,2580)=[];
test_data(:,2580)=[];

%delete from train and test data the column with dummy labels
train_data(:,2579)=[];
test_data(:,2579)=[];

%Transform to matrix form
totmatrix = double(totmatrix);
test_data = double(test_data);
train_data = double(train_data);
train_label = double(train_label);
test_label = double(test_label);



%% _________________ Calculation of Naive Bayes __________________________

%% Train Data

% spam and ham indices
spamIndex= find(train_label == 1);
hamIndex = find(train_label == 0);

%Num of documents in training data and number of words in all the messages
numdoc_train= size(train_data, 1)
numwords = size(train_data, 2)

% Calculate the prior, probability of being spam/ham
pSpam = length(spamIndex) / numdoc_train;
pHam= 1- pSpam

% Number of words in each message
wrds_per_mail = sum(train_data, 2);

% Total number of words for both spam and ham messages
totWrds_spam = sum(wrds_per_mail(spamIndex));
totWrds_ham= sum(wrds_per_mail(hamIndex));

% likelihood for spam, phi_(k|y=1) 
like_spam = (sum(train_data(spamIndex, :)) + 1) ./(totWrds_spam + numwords);

% likelihood for ham, phi_(k|y=0)
like_ham = (sum(train_data(hamIndex, :)) + 1)./(totWrds_ham + numwords);


%% Test

% Number of test documents for test data
numdoc_test = size(test_data, 1);

% The output vector will store the spam/ham prediction for the documents in our test set.
output = zeros(numdoc_test, 1);

% Calculate log p(x|y=1) + log p(y=1) and log p(x|y=0) + log p(y=0) for every document
% the prediction spam/ham is based on this comparison
log_a = test_data*(log(like_spam))' + log(pSpam);
log_b = test_data*(log(like_ham))'+ log(pHam);  
comparison = log_a > log_b;

% Wrong categorization for documents on the test set
wrong_cat= sum(xor(output, test_label))

%Percentage of wrong categorization on the test set
fraction_wrong = (wrong_cat/numdoc_test)*100
accuracy= 100-fraction_wrong
