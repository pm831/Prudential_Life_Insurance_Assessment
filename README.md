# Prudential Life Insurance Assessment
### Pujan Malavia

![prudential](https://user-images.githubusercontent.com/19572673/62430711-14dcd800-b6ee-11e9-9fb6-37f4715be0ee.png)

### Abstract:
In a one-click shopping world with on-demand everything, the life insurance application process is antiquated. Customers provide extensive information to identify risk classification and eligibility, including scheduling medical exams, a process that takes an average of 30 days.

The result? People are turned off. That’s why only 40% of U.S. households own individual life insurance. Prudential wants to make it quicker and less labor intensive for new and existing customers to get a quote while maintaining privacy boundaries.

By developing a predictive model that accurately classifies risk using a more automated approach, you can greatly impact public perception of the industry.

The results will help Prudential better understand the predictive power of the data points in the existing assessment, enabling us to significantly streamline the process. 

https://www.kaggle.com/c/prudential-life-insurance-assessment

### Industry:
Financial Services/Insurance

### Company Information:
Prudential plc is a British multinational life insurance and financial services company headquartered in London, United Kingdom. It was founded in London in May 1848 to provide loans to professional and working people.

Prudential has 26 million life customers. It owns Prudential Corporation Asia, which has leading insurance and asset management operations across 14 markets in Asia, Jackson National Life Insurance Company, which is one of the largest life insurance providers in the United States, and M&GPrudential, a leading savings and investments business serving customers in the UK and Europe 

Prudential has a primary listing on the London Stock Exchange and is a constituent of the FTSE 100 Index. Prudential has secondary listings on the Hong Kong Stock Exchange, New York Stock Exchange and Singapore Exchange. 

https://en.wikipedia.org/wiki/Prudential_plc

https://www.prudential.com/

### Use Case:

Developing a predictive model that accurately classifies risk using a more automated approach

### Initial Dataset:
train.csv - the training set, contains the Response values
test.csv - the test set, you must predict the Response variable for all rows in this file sample_submission.csv - a sample submission file in the correct format

### Tool:
R

### Data

In this dataset, you are provided over a hundred variables describing attributes of life insurance applicants. The task is to predict the "Response" variable for each Id in the test set. "Response" is an ordinal measure of risk that has 8 levels.

### Data Fields:

Id	A unique identifier associated with an application.

Product_Info_1-7	A set of normalized variables relating to the product applied for

Ins_Age		Normalized age of applicant

Ht	Normalized height of applicant

Wt	Normalized weight of applicant

BMI	Normalized BMI of applicant

mployment_Info_1-6	A set of normalized variables relating to the employment history of the applicant.

InsuredInfo_1-6	A set of normalized variables providing information about the applicant.

Insurance_History_1-9	A set of normalized variables relating to the insurance history of the applicant.

Family_Hist_1-5	A set of normalized variables relating to the family history of the applicant.

Medical_History_1-41	A set of normalized variables relating to the medical history of the applicant.

Medical_Keyword_1-48	A set of dummy variables relating to the presence of/absence of a medical keyword being associated with the application.

Response	This is the target variable, an ordinal variable relating to the final decision associated with an application

### Communication of Results to Business Partner:
To a business partner, I would explain that the xgBoost (all else equal) is an efficient and easy to use algorithm which delivers high performance and accuracy as compared to other algorithms.

### Future Work:
Continue to do hyperparameter tuning of the model and creating new features/removing old features to help increase the prediction accuracy of the model

Try other types of models to see if the accuracy rate improves

More data visualization/patterns within the dataset (external sources) that can lead to more insights and decision-making from a business perspective
