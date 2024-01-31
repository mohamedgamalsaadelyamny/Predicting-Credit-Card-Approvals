#### Credit Cards Project
## METHODS USED
D E C I S I O N T R E E , S V M , N A Ï V E B A Y E S , X G B O O S T, P E R C E P T R O N
CONCLUSION
C O M P A R I N G R E S U LT S F O R E A C H M O D E L
## DATA INTRODUTION
OUR CREDI T CARD APPROVAL S DATAS E T I S A COL L ECT ION OF
DATA THAT RE VOLVE S AROUND THE PROCE S S OF AS S E S S ING AND
APPROVING CREDI T CARD APPL ICAT IONS . THI S DATAS E T I S
VALUABL E FOR BUI LDING PREDICT IVE MODE L S AIMED AT
AUTOMAT ING THE DECI S ION-MAKING PROCE S S IN CREDI T CARD
APPROVAL S Y S T EMS . THE INFORMAT ION CONTAINED IN THE
DATAS E T T Y PICAL LY INCLUDE S VARIOUS FEATURE S AND
AT TRIBUT E S RE LAT ED TO THE APPL ICANT S ' FINANCIAL PROFI L E S ,
PERSONAL DE TAI L S , AND CREDI T HI S TORY.
## FEATURES:
GENDER, AGE , DEBT, MARRI ED, BANKCUS TOMER, INDUS TRY,
E THNICI T Y, Y EARS EMPLOY ED, PRIORDE FAULT, EMPLOY ED,
CREDI T SCORE , DRIVERS L ICENS E , CI T I ZEN, ZIPCODE , INCOME
## TARGET VARIABLE:
THE DATAS E T T Y PICAL LY INCLUDE S A TARGE T VARIABL E
INDICAT ING WHE THER THE CREDI T CARD APPL ICAT ION WAS
APPROVED OR DENI ED. THI S BINARY OUTCOME VARIABL E I S
CRUCIAL FOR BUI LDING A CLAS S I FICAT ION MODE L .
## OBJECTIVE:
THE PRIMARY OBJECT IVE OF WORKING WI TH THI S DATAS E T I S TO
DE VE LOP PREDICT IVE MODE L S THAT CAN AUTOMAT ICAL LY
AS S E S S NEW CREDI T CARD APPL ICAT IONS AND MAKE INFORMED
DECI S IONS ABOUT APPROVAL OR RE JECT ION. THI S CONTRIBUT E S
TO THE E FFICI ENCY AND ACCURACY OF THE CREDI T APPROVAL
PROCE S S .
## METHODS USED
DECISION TREE, SVM, NAÏVE BAYES, PERCEPTRON


## DECISION TREES
Credit Card Approval Prediction Using Decision Tree Classifier
Introduction:
The provided Python code demonstrates the process of building a decision tree model to
predict credit card approval. The analysis involves importing necessary libraries, exploring
the dataset, handling missing data, converting categorical features, and training a decision
tree classifier.

Step 1: Importing Libraries
The code begins by importing essential libraries, including pandas for data manipulation,
NumPy for numerical operations, and matplotlib/seaborn for data visualization. %matplotlib
inline is used for inline plotting.

Step 2: Reading the Data
The credit card approvals dataset is loaded into a pandas data frame (df) using the
pd.read_csv function. The first few rows of the dataset are displayed using df.head() to
provide an initial overview.

20XX 7
Step 3: Exploratory Data Analysis (EDA)

Missing Data Analysis:
A seaborn heatmap is generated to visualize missing data in the
dataset. This helps identify any gaps in the data.

20XX Pitch Deck 8
Visualizing Approval Status:
Seaborn count plots are used to visualize the distribution of approved and
rejected credit card applications.

20XX Pitch Deck 9
Additional count plots are created to explore approval status based on
gender and ethnicity.

20XX 10
A histogram is plotted for the 'Credit Score' column.

Step 4: Converting Categorical Features
Categorical features such as 'Industry', 'Ethnicity' , and 'Citizen' is converted into
dummy variables using the pd.get_dummies function. The original
categorical columns are dropped, and the dummy variables are
concatenated with the data frame.

Step 5: Building a Decision Tree Model
A decision tree is a supervised machine learning algorithm that can be used
for classification tasks. It works by recursively splitting the dataset into
subsets based on the most significant attribute at each node of the tree. The
process continues until a stopping condition is met, such as reaching a
maximum depth or having a subset with only one class

Train-Test Split:
The dataset is split into training and testing sets using the train_test_split
function from sklearn.model_selection.

Training the Decision Tree Model:
A Decision Tree Classifier is initialized and trained using the training data.

Step 6: Model Evaluation

Predictions:
The trained model is used to make predictions on the test set.
Evaluation Metrics:
The classification report and confusion matrix from sklearn.metrics are
used to evaluate the model's performance. Precision, recall, and F1-
score are calculated for each class (approved and rejected), providing a
comprehensive assessment.
Classification report:
precision recall f1-score support
0 0.79 0.81 0.80 110
1 0.78 0.75 0.76 97
accuracy 0.78 207
macro avg 0.78 0.78 0.78 207
weighted avg 0.78 0.78 0.78 207

confusion matrix:

[[89 21]
[24 73]]
Conclusion:
The decision tree model achieves an accuracy of approximately 78%, as
indicated by the evaluation metrics. Further analysis or optimization
steps could be taken based on the insights gained from EDA and model
evaluation.

## SUPPORT VECTOR MACHINES

✓ Introduction to SVMs
✓ Data Loading and Preprocessing
✓ Exploratory Data Analysis (EDA)
✓ Model Training
✓ Evaluation Metrics
✓ Results and Analysis
✓ Conclusion

Agenda

All code details are included in the SVM notebook
Support Vector Machines (SVMs) are a type of supervised
machine learning algorithm used for classification and
regression tasks. They are widely used in various fields,
including pattern recognition, image analysis, and natural
language processing.
SVMs work by finding the optimal
hyperplane that separates data
points into different classes.
Maximizing the margin

Data Loading and Preprocessing:

✓ Loading data
▪ Link
✓ Knowing the Data
▪ Structure
▪ Statistics
✓ Data processing
▪ Splitting Data
▪ Handling missing values
▪ Encoding categorical variables

Exploratory Data Analysis (EDA)

✓ Features Correlation
✓ Class Distribution of target

The correlation matrix of the data features shows that:

PriorDefault and Approved : There is a significant positive correlation of 0.72,
indicating that applicants have no prior defaults are more likely to get approved.
CreditScore, Employed, and Approved : CreditScore has notable positive
correlations with both Employed (0.57) and Approved (0.46), suggesting that
higher credit scores are associated with employment status and approval rate.
BankCustomer and Married : There is a strong positive correlation of 0.99,
indicating these two features almost always occur together.'‘’
Features with high correlation with the target variable (Approved) like
PriorDefault, CreditScore, and Employed could be given more importance.
However, features that are highly correlated with each other, like BankCustomer
and Married, might make the model unstable
So, I will exclude one of the two features either BankCustomer or Married.
Married feature excluded which has low correlation with the
target variable (Approved)
PriorDefault & CreditScore are dominant features
Model Training:

✓ Using SVC model
✓ Choosing kernel, Linear
✓ Feature Scaling

Evaluation Metrics:
✓ Accuracy
✓ Precision
✓ Recall
✓ F1 score
✓ Confusion matrix
Metric Value(%)
Accuracy 80.
Precision 79.

Recall 82.
F1 score 81.
✓ Overall Conclusion:

The SVM model demonstrates good accuracy on both the training and test sets
The confusion matrix shows a balanced performance
Precision, recall, and F1 score provide a balanced assessment of the model’s
ability to identify credit card approvals.
✓ In conclusion, based on the provided metrics, the SVM model appears to be
effective in predicting credit card approvals.

## NAIVE BAYES CLASSIFIER

Naïve bayes classifier is a probabilistic Generative Model that Learns likelihood and prior probabilities then
calculate theclasses condition probability

Defining the likelihood of the data as the probability of observing the data under the assumptions:
Gaussian distribution for class-conditional densities
Shared covariance matrix for all classes (so that the posterior probability would be a linear function of the input data)
Input data points independence (so that the likelihood function would be the product of the joint probabilitiesof the
input data and their given classes)
20XX 24
The correlation matrix of the data features shows that there is a high correlation between Married and Bank Customer
Features, so I discarded the Married -since it has lower correlation with the approval outcome- so that the 3rd
assumption becomes valid

The results were

85.3% training Accuracy
85.8% training F-score
87.6% testing Accuracy
87.7% testing F-score
This is a plot for classes conditional probabilities and the decision boundary for the two highest correlation features
Prior Default and Employed
20XX 26
To possible improve the accuracy, I allowed each class to have its own covariance matrix, so the quadratic terms of
the input data won't cancel each other according to pattern recognition book by Christopher bishop page 199, and
we would have a quadratic decision boundary possible improving the separationaccuracy.
Hence, the final equations would have a different forms.
20XX 27
The results were unfortunately not any better (maybe if we discard the third assumption of input data independence)

80.0% training Accuracy
84.0% training F-score
84.0% testing Accuracy
86.5% testing F-score
This is a plot for classes conditional probabilities and the decision boundary for the two highest correlation features
Prior Default and Employed


## perceptron
In our data wemeasure the similarity
between the labels ,Anddrop features with
mi score low than .01
And makingscaling for the features
Mi score
Income 0.303647
PriorDefault 0.295079
ZipCode 0.201648
CreditScore 0.167950
Employed 0.108329
Industry 0.075664
YearsEmployed 0.061593
Ethnicity 0.032413
BankCustomer 0.018488
Married 0.016834
Debt 0.011751
Citizen 0.006957
DriversLicense 0.000500
Gender 0.000418

scaling the features
sklearn.preprocessing.StandardScaler(*, copy=True,with_mean=True,
with_std=True)
Standardize features by removing the mean and scaling to unit
variance.
The standard score of a samplexis calculated as:
z = (x - u) / s
whereuis the mean of the training samples or zero
ifwith_mean=False, andsis the standard deviation of the training
samples or one ifwith_std=False.
accuracy for training before scaling: 0.7137681159420289
Accuracy for testing before scaling: 0.7028985507246377

accuracy for training After scaling: 0.9981884057971014
Accuracy for testing After scaling: 0.9855072463768116

Perceptron Learning:
In our case : this is a binary linear classifier that our

model divides the input into two classes by learning a

separating hyperplane iteratively.

The data are linearly separable because the accuracy

of training is close to 100% and our model finds a

separating hyperplane and converges

CONCLUSION
20XX 39

DT overfits the data as expected but XGBoot reduces this overfitting and improves the
accuracy so that it is close to MLE results
SVM and Naïve bayes solution are very close, however Naïve bayes is the only model that
has testing accuracy better than the training accuracy due to its nature, Naïve bayes
models the input -likelihood- very accuratelyso that it generalizes so well on the testing
data
Perceptron results are the highest on this particular data
