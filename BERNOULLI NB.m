%____________________________ MACHINE LEARNING ___________________________

% Changing working directory
%cd 'C:\Users\Fede\Documents\City University of London\MSc - Data Science\Machine Learning\Coursework';
data = readtable('spam.csv');
% convert table to dataset
ds = table2dataset(data); 

% Emptying Columns for useless variables
ds(:,'Var5') = [];
ds(:,'Var4') = [];

% Change Variables Names
ds.Properties.VarNames{1} = 'spam';
ds.Properties.VarNames{2}= 'txt';
ds.Properties.VarNames{3}= 'dummy';

% Convert variables to dummy 1 for ham and 2 for spam
ds.dummy = grp2idx(ds.spam);
% Split dataset in 2 variables
not_spam = ds(ds.dummy==1,:);
spam_ds = ds(ds.dummy==2,:);

%%
%           ___________________ SPAM Word Cloud ______________________

% Create Variable for spam classified messages
text = spam_ds.txt;
% Erase Punctuation 
cleanTextData = erasePunctuation(text);
% All lower case
cleanTextData = lower(cleanTextData);
% Remove stowords
dirttxt = tokenizedDocument(cleanTextData);
dirttxt = removeWords(dirttxt,stopWords);
% Remove Long and short words
dirttxt = removeShortWords(dirttxt,2);
dirttxt = removeLongWords(dirttxt,15);
%cleanDocuments = normalizeWords(cleanDocuments);
spambag = bagOfWords(dirttxt);
spambag = removeInfrequentWords(spambag,2);
spambag = bagOfWords(dirttxt);
wordcloud(spambag);

%%
%        ___________________ NOT SPAM Word Cloud ______________________

% Create Variable for spam classified messages
text = not_spam.txt;
% Erase Punctuation 
cleanTextData = erasePunctuation(text);
% All lower case
cleanTextData = lower(cleanTextData);
% Remove stopwords
clean_spam = tokenizedDocument(cleanTextData);
clean_spam = removeWords(clean_spam,stopWords); % Removing stopwords
clean_spam = removeShortWords(clean_spam,2); % Removing excessively short words
clean_spam = removeLongWords(clean_spam,15); % Removing excessively long words
%cleanDocuments = normalizeWords(cleanDocuments); % Normalizing words...
cleanBag = bagOfWords(clean_spam); % Creating bag of words
cleanBag = removeInfrequentWords(cleanBag,2); % Removing infrequent words
cleanBag = bagOfWords(clean_spam); % Producing the bag of words
wordcloud(cleanBag); % Plotting the cloud.

%%
spamT = T(dirttxt); % Produce a table with spam words
nospamT = T(clean_spam); % Produce a table of non-spam words

J = innerjoin(nospamT, spamT,'Keys','Words'); % Join two tables per Words 

for i=1:length(J.Words) % Making Dummy Variables for words that are both considered Spam and Not_Spam
    if (J.NumOccur_nospamT(i) >= 50 && J.NumOccur_spamT(i) >= 50)
        dummina(i,1) = 1;
    elseif (J.NumOccur_nospamT(i) < 50 && J.NumOccur_spamT(i) < 50)
        dummina(i,1) = 0;
    else
    end
end

J.dummy = dummina; % Attach dummy to table
J = sortrows(J,'dummy', 'descend'); % sort to have all the ones on top

for i=1:length(J.Words) % visualize our target and create pattern var to clean the text
    if (J.dummy(i) == 1)
        disp(J.Words(i,1)); %  identify position
        pat_txt(i,1) = J.Words(i,1); % pattern variable
    else
    end
end

pat_txt = string(pat_txt);
dirttxt = removeWords(dirttxt,pat_txt);
dirttxt = removeWords(dirttxt,stopWords);
spamT = T(dirttxt); % Use function to produce a new table clean of repeated values
spamT(1:30,:)

% Visualize the new cloud
spambag = bagOfWords(dirttxt);
spambag = removeInfrequentWords(spambag,2);
spambag = bagOfWords(dirttxt);
wordcloud(spambag);


%%
% Create variables for the most used word for spamming
% We decided to divide the most used words by category:
% Winner Category

pat_win = ["win", "awarded", "won", "award", "prize", "claim", "congratulations", "selected", "gift"];
winner = contains(ds.txt, pat_win);


% Attention-related category
pat_attention = ["urgent", "1st", "stop", "latest", "cash"]; % Making an array of all the attention-related words.
attention = contains(ds.txt, pat_attention); % create a variable "attention" which 
                                             % is a vector of 1 and 0 for
                                             % every match found in the
                                             % text. Do same thing for the
                                             % others.

% Mobile/Phone Related category:
pat_mobile = ["nokia", "txt", "phone", "reply", "chat", "line", "tone", "mobile", "mins", "camera", "msg"]; 
mobile = contains(ds.txt, pat_mobile); 

% Ads/offer Related words:
pat_ads = ["customer", "service", "services", "guaranted", "contact", "apply", "code", "entry", "unsubscribe"]; % 
ads = contains(ds.txt, pat_ads);


% Adult/Dating Related Words:
pat_adult = ["dating", "sexy", "sex", "xxx", "love"];
adult = contains(ds.txt, pat_adult);

%å character
strange_a= ['å'];
strange_a= contains(ds.txt, strange_a);


%Attaching Variable to the main dataset
ds.winner = winner;
ds.attention = attention;
ds.mobile = mobile;
ds.adult = adult;
ds.ads = ads;
ds.strange_a= strange_a; 




%% TRAIN SET DATA
% Dividing dataset 
Q = length(ds);
Q = round(Q);
trainRatio = 0.8;
testRatio = 0.2;
[trainInd,testInd] = dividerand(Q,trainRatio,testRatio);


for i = 1:length(trainInd)
    v = trainInd(i);
    train_test(v,1)=1;
end

ds.train = train_test;
train_data = ds(ds.train==1,:);
test_data = ds(ds.train==0,:);

%% Decision tree model:
X = table(train_data.ads, train_data.adult, train_data.attention, train_data.strange_a, train_data.mobile, train_data.winner,'VariableNames', {'ads', 'adult', 'attention', 'strange_a', 'mobile', 'winner'});
Y = [train_data.spam];
Mdl = fitctree(X,Y);
view(Mdl,'mode','graph')

cvmodel = crossval(Mdl);
L = kfoldLoss(cvmodel)



%%     ______________ NAIVE BAYES with BERNOULLI MODEL DOCUMENT  ________________



%% PREPROCESSING DATA 

% For Naive Bayes we will use the same database of decision tree
%I only change the name of it so to avoid confusion when I will change some 
% of its features. The document that will be processed is a 
%'Bernoulli style document (Shimodaira, H. 2015)

ds_b=ds;

%delete 'spam' column and 'txt' column because they won't fit into a matrix
%(they have string values)
ds_b.spam=[];
ds_b.txt=[];

%recode dummy variable: spam=1 ham=0 (previously spam=2 ham=1)
ds_b.dummy(ds_b.dummy== 1) = 0;
ds_b.dummy(ds_b.dummy== 2) = 1;

%divide into test/train set
train_ds = ds_b(ds_b.train==1,:);
test_ds = ds_b(ds_b.train==0,:);

%delete train column 
ds_b.train=[];
train_ds.train=[];
test_ds.train=[];

%Change class from dataset to matrix/vector
nbM = double(ds_b);
train_databM = double(train_ds); 
test_databM = double(test_ds); 

%% VARIABLE PREPARATION 

% ham/spam vector
y0 = nbM(:,1);
y1 = train_databM(:,1);    
y2 = test_databM(:,1);   

% categories of predictors (adult, winner...) 
x0= nbM(:,2:end);
x1 = train_databM(:,2:end); 
x2= test_databM(:, 2:end);   

%% RUN NAIVE BAYES WITH BERNOULLI DISTRIBUTION

%%%Train Naive Bayes Classifiers Using Multinomial Predictors 
%train the classifier
Mdl = fitcnb(x1,y1,'Distribution','mn');

%Assess the in-sample performance of Mdl by estimating the misclassification error.
isGenRate = resubLoss(Mdl,'LossFun','ClassifErr')
%isGenRate = 0.13 means the accuracy is 87%

%Now use the test data for assessing Classifier Performance 
%and determine whether the algorithm generalizes.
oosGenRate = loss(Mdl,x2, y2)
%oosGenRate = 0.129 means the result is generalizable (still 87/88% accuracy by and large)


%% CROSS VALIDATION

Mdl_1 = fitcnb(x0,y0,'Distribution','mn');
CrossValiMdl = crossval(Mdl_1)
kfoldLoss(CrossValiMdl)

% ans= 0.1301, accuracy confirmed to be by and large 87%


%% NAIVE BAYES OPTIMIZATION (graphs) 

classNames = {'1', '0'};
rng default
Mdl_opt = fitcnb(x0,y0,'Distribution','mn','ClassNames',classNames,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'))


%% Predict labels using naive Bayes classification model

%Train a naive Bayes classifier and specify to holdout 20% of the data for a test sample 
%[This method of splitting train/test data is much faster]. 
rng(1);
            
CVMdl = fitcnb(x0,y0, 'Distribution','mn', 'Holdout',0.20,'ClassNames',{'1', '0'});
CMdl = CVMdl.Trained{1};          
testIdx = test(CVMdl.Partition); 
XTest = x0(testIdx,:);
YTest = y0(testIdx);

%Label the test sample observations. Display the results for a random set of 10 observations in the test sample.

idx = randsample(sum(testIdx),10);
label = predict(CMdl,XTest);
table(YTest(idx),label(idx),'VariableNames',...
    {'TrueLabel','PredictedLabel'})

%% Estimate Posterior Probabilities and Misclassification Costs

n = size(x0,1);
newInds = randsample(n,10);
inds = ~ismember(1:n,newInds);
XNew = x0(newInds,:);
YNew = y0(newInds);

Mdl = fitcnb(x0(inds,:),y0(inds),'Distribution','mn','ClassNames',{'1', '0'});
CMdl = compact(Mdl);
whos('Mdl','CMdl')

CMdl.ClassNames
[labels,PostProbs,MisClassCost] = predict(CMdl,XNew);
table(YNew,labels,PostProbs,'VariableNames',{'TrueLabels','PredictedLabels','PosteriorProbabilities'})
MisClassCost

