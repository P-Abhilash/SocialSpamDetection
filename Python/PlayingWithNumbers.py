import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

print('##### Data Info ##### ')

df_bots = pd.read_csv('/content/SocialSpamDetection/CSVs/bots_data.csv', sep=",", encoding='latin1')
df_nonbots = pd.read_csv('/content/SocialSpamDetection/CSVs/nonbots_data.csv', sep=",", encoding='latin1')
# print(df_bots.head(5))

df = pd.concat([df_bots, df_nonbots], ignore_index=True)
df.fillna('?', inplace=True)
print('total: {}'.format(df.shape))

# add a column 'nb_guess' with a Naive Bayes classification of the description
df['nb_guess'] = pd.read_csv('/content/SocialSpamDetection/CSVs/nb_guess.csv', header=None)

# add a column 'svm_guess' with a SVM classification of the description
df['svm_guess'] = pd.read_csv('/content/SocialSpamDetection/CSVs/svm_guess.csv', header=None)

# removing unnecessary columns. keeping only numbers for this part
df = df.drop(['id', 'id_str', 'url', 'default_profile', 'default_profile_image', 'screen_name', 'location',
              'has_extended_profile', 'status', 'lang', 'description', 'created_at', 'name'], 1)
print(df.columns)

# split the dataset into 80/20
split = np.random.rand(len(df)) < 0.8
train_df = df[split]
test_df = df[~split]

# print('train: {}, test: {}'.format(train_df.shape, test_df.shape))

## Random Forests
print('\n----- Random Forests -----')
X = train_df.drop('bot', 1)
Y = train_df['bot']
from sklearn.ensemble import RandomForestClassifier

clf_RF = RandomForestClassifier(criterion='entropy')
clf_RF.fit(X, Y)
print('accuracy on training data: {}'.format(clf_RF.score(X, Y)))
print('accuracy on test data: {}'.format(clf_RF.score(test_df.drop('bot', 1), test_df['bot'])))
print()
plot_confusion_matrix(clf_RF, X, Y)
plt.title("RandomForestClassifier")


## Decision Tree
# create the classifier
print('\n----- Decision Trees -----')
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


clf_dt = DecisionTreeClassifier(criterion='entropy')
clf_dt = clf_dt.fit(train_df.drop('bot', 1), train_df['bot'])

# testing the classifier
predict_train_dt = clf_dt.predict(train_df.drop('bot', 1))
predict_test_dt = clf_dt.predict(test_df.drop('bot', 1))

# print accuracies
print('accuracy on training data: {}'
      .format(accuracy_score(train_df['bot'], predict_train_dt)))
print('accuracy on test data: {}'
      .format(accuracy_score(test_df['bot'], predict_test_dt)))
print()


## Linear Regression doesnt make sense because well, either bot or not.

## Logistic Regression

print('\n----- Logistic Regression -----')
from sklearn.linear_model import LogisticRegression

X = train_df.drop('bot', 1)
y = train_df['bot']

    # turn down tolerance for short training time
clf_l1_LR = LogisticRegression( penalty='l1', tol=0.01, solver='liblinear')
clf_l1_LR.fit(X, y)

print('accuracy on training data: %.4f' % clf_l1_LR.score(X, y))

print('accuracy on test data: %.4f' % clf_l1_LR.score(test_df.drop('bot', 1), test_df['bot']))

print()
plot_confusion_matrix(clf_l1_LR, X, Y)
plt.title("Logistic Regression")


## Naive Bayes - Bernoulli and Multnomial
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

X = train_df.drop('bot', 1)
Y = train_df['bot']

clf_BNB = BernoulliNB()
clf_MNB = MultinomialNB()

clf_BNB.fit(X, Y)
clf_MNB = clf_MNB.fit(X, Y)

clf_BNB_score_train = clf_BNB.score(X, Y)
clf_MNB_score_train = clf_MNB.score(X, Y)
clf_BNB_score_test = clf_BNB.score(test_df.drop('bot', 1), test_df['bot'])
clf_MNB_score_test = clf_MNB.score(test_df.drop('bot', 1), test_df['bot'])

print('\n----- Naive Bayes -----')
print('Bernoulli NB accuracy on training data: {}'.format(clf_BNB_score_train))
print('Multinomial NB accuracy on training data: {}'.format(clf_MNB_score_train))
print('Bernoulli NB accuracy on test data: {}'.format(clf_BNB_score_test))
print('Multinomial NB accuracy on test data: {}'.format(clf_MNB_score_test))
print()


from matplotlib import pyplot as plt
from sklearn import metrics

fpr_mnb, tpr_mnb, _ = metrics.roc_curve(test_df['bot'], clf_MNB.predict(test_df.drop('bot', 1)))
fpr_bnb, tpr_bnb, _ = metrics.roc_curve(test_df['bot'], clf_BNB.predict(test_df.drop('bot', 1)))
fpr_dt, tpr_dt, _ = metrics.roc_curve(test_df['bot'], clf_dt.predict(test_df.drop('bot', 1)))
fpr_RF, tpr_RF, _ = metrics.roc_curve(test_df['bot'], clf_RF.predict(test_df.drop('bot', 1)))

plt.figure(1)
plt.plot(fpr_mnb, tpr_mnb, label='MNB')
plt.plot(fpr_bnb, tpr_bnb, label='BNB')
plt.plot(fpr_dt, tpr_dt, label='DT')
plt.plot(fpr_RF, tpr_RF, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()