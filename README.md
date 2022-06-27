# Rainfall prediction model using ML

## 1. Abstract
Rainfall has been a major concern these days. Weather conditions have been changing for the last 10 years. This has paved the way for drastic changes in patterns of rainfall.  

Rainfall prediction model mainly based on artificial neural networks are not the mainstream. This project does a comparative study of four rainfall prediction approaches and finds the best one. The present technique to predict rainfall doesn’t work well with the complex data present. The approaches which are being used now-a-days are statistical methods and numerical methods, which don’t work accurately when there is any non-linear pattern. Existing system fails whenever the complexity of the datasets which contains past rainfall data increases. Henceforth, to find the best way to predict rainfall, study of both Machine Learning and Deep Learning methods must be performed and the approach which gives better accuracy should be used in prediction.  

Rainfall can be considered as the pillar for the primary source of most of the economy of our country, i.e. Agriculture. To do a proper investment on agriculture, a proper estimation of rainfall is needed. Irregular heavy rainfall may lead to the destruction of crops, heavy floods, that can cause harm to human life. It is important to exactly determine the rainfall for effective use of water resources, crop productivity, and pre-planning of water structures.  

People in coastal areas are in high risk of heavy rainfall and floods, so they should be aware of the rainfall much earlier so that they can plan their stay accordingly. For areas which have less rainfall and faces water scarcity should have rainwater harvesters, which can collect the rainwater. To establish a proper rainwater harvester, rainfall estimation is required. Weather forecasting is the easiest and fastest way to get a greater outreach. This research work can be used by all the weather forecasting channels, so that the prediction news can be more accurate and can spread to all parts of the country.  


## 2. Introduction to the Rainfall Prediction Model

The objective of the Rainfall prediction model is basically to predict the approximate amount of rainfall an area will receive in the next month using the rainfall amount of current month as input. It also gives us a measure of the trend or pattern the rainfall will follow. Which is very helpful for the farmers, as It will give appropriate time for them to prepare for, in-case low rainfall is predicted. 
This will also help governmental bodies to keep track of areas which may face less rainfall and may need assistance in form of better water supply, creation of hand pumps, etc.  

For this, the past rainfall data have been gathered  and arranged it in a time series dataset. The dataset is then used to create Prediction models using 4 different approaches. This is done to explore a range of algorithms that are used to create time-series forecasting models.
While any of them could have been used to create the required model, best practice is to test a few of the prominent ones over a subset of the dataset and then compare them to get a measure of which algorithm is best of the dataset. And then use it to create model for the whole dataset.

## 3. System Requirement and Specification
System requirements are the configuration that a system must have in order for a hardware or software application to run smoothly and efficiently. Failure to meet these requirements can result in installation problems or performance problems. The former may prevent a device or application from getting installed, whereas the latter may cause a product to malfunction or perform below expectation or even to hang or crash. System requirements are also known as minimum system requirements.  

The system must have following requirement:  

- Central Processing Unit (CPU) – Intel core i5 6th  generation processor or higher. An AMD equivalent processor will also be optimal.
- Ram: 4GB minimum,  8 GB or higher is recommended.
- Hard Disk: 250 GB.
- Graphics Processing Unit(GPU) – NVIDIA GeForce GTX960 or higher. AMD GPUs are not  able to perform  deep perform regardless. 
- OS: Ubuntu or Microsoft Windows 10.I recommend updating Windows 10 to the latest version before proceeding forward.
- Programming: Python 3.6 or above version and related library files.
- Environment: jupyter notebook 

## 4. Literature

### Linear Regression

Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables they are considering, and the number of independent variables getting used.  


![image.png](attachment:82e3eda3-dc0e-4e94-862a-ee3371044e0b.png)

Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x).  
So, this regression technique finds out a linear relationship between x (input) and y(output). Hence, the name is Linear Regression.  
In the figure(fig. 1) above, X (input) is the work experience and Y (output) is the salary of a person. The regression line is the best fit line for our model.  

Hypothesis function for Linear Regression :  
![image.png](attachment:e7fe94c2-8006-4f97-b955-af2a288a29e0.png)

While training the model we are given :  
x: input training data (univariate – one input variable(parameter))  
y: labels to data (supervised learning)  

When training the model – it fits the best line to predict the value of y for a given value of x. The model gets the best regression fit line by finding the best θ1 and θ2 values.  
θ1 : intercept  
θ2: coefficient of x  

Once we find the best θ1 and θ2 values, we get the best fit line. So when we are finally using our model for prediction, it will predict the value of y for the input value of x.  

### How to update θ1 and θ2 values to get the best fit line ?
### Cost Function (J):

By achieving the best-fit regression line, the model aims to predict y value such that the error difference between predicted value and true value is minimum. So, it is very important to update the θ1 and θ2 values, to reach the best value that minimize the error between predicted y value (pred) and true y value (y).  

![image.png](attachment:cf25fbfe-0a77-45d2-9cbe-05efc08ecc34.png)  

![image.png](attachment:0f2592b7-90ab-43d7-92fe-9ee4b35d4b5c.png)

Cost function(J) of Linear Regression is the Root Mean Squared Error (RMSE) between predicted y value (pred) and true y value (y).

### Gradient Descent:
Gradient Descent is an optimization algorithm used for minimizing the cost function in various machine learning algorithms. It is basically used for updating the parameters of the learning model. To update θ1 and θ2 values in order to reduce Cost function (minimizing RMSE value) and achieving the best fit line the model uses Gradient Descent. The idea is to start with random θ1 and θ2 values and then iteratively updating the values, reaching minimum cost.  

![image.png](attachment:872e2312-cc66-44fd-9c71-78e9505e1684.png)  

- θj     : Weights of the hypothesis.
- hθ(xi) : predicted y value for ith input.
- j     : Feature index number (can be 0, 1, 2, ......, n).
- α     : Learning Rate of Gradient Descent.


### Regularization in Machine Learning

Overfitting(fig. 3) is a phenomenon that occurs when a Machine Learning model is constraint to training set and not able to perform well on unseen data.  

![image.png](attachment:4fd183f5-8706-446a-9c28-b26587201353.png)  

Regularization works by adding a penalty or complexity term to the complex. The commonly used regularization techniques are :   
1. L1 regularization(Lasso regression) :- It is calculated by multiplying  the sum of absolute weight of each individual feature to lambda which is then added as penalty to the cost function.  
2. L2 regularization(Ridge regression) :- It is calculated by multiplying  the sum of squared weight of each individual feature to lambda which is then added as penalty to the cost function.  

Ridge regression is mostly used to reduce the overfitting in the model, and it includes all the features present in the model. It reduces the complexity of the model by shrinking the coefficients.  
Lasso regression helps to reduce the overfitting in the model as well as feature selection.  

### Elastic Net Regression

Elastic Net(fig. 4) is a regression method that performs variable selection and regularization both simultaneously. Elastic net is a popular type of regularized linear regression that combines two popular penalties, specifically the L1 and L2 penalty functions.  

![image.png](attachment:c500f71f-3fb7-47f7-be0b-b0bfa1b810ec.png)

### Long short-term memory(LSTM)
Long short-term memory(fig. 5) (LSTM) is an artificial recurrent neural network (RNN) used in the field of deep learning. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. The functions of the three types of gates are:  
- forget gate – controls how much information the memory cell will receive from the memory cell from the previous step
- update (input) gate – decides whether the memory cell will be updated. Also, it controls how much information the current memory cell will receive from a potentially new memory cell.
- output gate – controls the value of the next hidden state  

![image.png](attachment:aeb8181e-a938-42eb-981a-9d3ae76410a0.png)  

LSTM networks are used to classifying, processing and making predictions based on time series data. LSTM is a special type of recurrent neural network. Specifically, this architecture is introduced to solve the problem of vanishing and exploding gradients. In addition, this type of network is better for maintaining long-range connections, recognizing the relationship between values at the beginning and end of a sequence.
### Stacked LSTM
A Stacked LSTM architecture can be defined as an LSTM model comprised of multiple LSTM layers. An LSTM layer above provides a sequence output rather than a single value output to the LSTM layer below. Specifically, one output per input time step, rather than one output time step for all input time steps.
### Bidirectional LSTM
A Bidirectional LSTM, or BiLSTM, is a sequence processing model that consists of two LSTM layers: one taking the input in a forward direction, and the other in a backwards direction. BiLSTMs effectively increase the amount of information available to the network, improving the context available to the algorithm.  

Unlike standard LSTM, the input flows in both directions, and it’s capable of utilizing information from both sides. It’s also a powerful tool for modeling the sequential dependencies between inputs and outputs in both directions of the sequence.  

![image.png](attachment:5706230e-151a-4086-8097-7da3eb880842.png)  

In summary, BiLSTM adds one more LSTM layer, which reverses the direction of information flow. Briefly, it means that the input sequence flows backward in the additional LSTM layer. Then we combine the outputs from both LSTM layers in several ways, such as average, sum, multiplication, or concatenation.This type of architecture has many advantages in real-world problems. The main reason is that every component of an input sequence has information from both the past and present. For this reason, BiLSTM can produce a more meaningful output, combining LSTM layers from both directions.  


### XGBoost 
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving  small-to-medium structured/tabular data, decision tree based algorithms are considered best-in-class right now.  

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. It is a powerful approach for building supervised regression models. The validity of this statement can be inferred by knowing about its (XGBoost) objective function and base learners.
The objective function contains loss function and a regularization term. It tells about the difference between actual values and predicted values, i.e how far the model results are from the real values. The most common loss functions in XGBoost for regression problems is reg:linear, and that for binary classification is reg:logistics.  

Ensemble learning involves training and combining individual models (known as base learners) to get a single prediction, and XGBoost is one of the ensemble learning methods. XGBoost expects to have the base learners which are uniformly bad at the remainder so that when all the predictions are combined, bad predictions cancels out and better one sums up to form final good predictions.

## 5. Methodology
The proposed methodology of this project is rather simple. Time series data has been collected  from trusted data sources. Then using various visualization techniques the data has been evaluated to check for its accuracy and to find the various trends  present in the dataset. Thereafter, having confirmed the trends as well as the dependencies of various features present in the dataset, the process for creation of model from the data has been selected. In this project one-step forecasting method has been used, which means given the rainfall in the last  month the model tries to predict the rainfall in the next month.  

As the dataset is too big it has been divided into various subsets according to sub-divisions and only one sub-division(area) data has been selected. In the next step  data has been converted into the correct format to fit it into the 4 models(2 machine learning models and 2 deep learning models). The processed data has been separated into training set and testing set then  4 models has been trained using the training dataset. Then using the test dataset the 4 approaches used to create models has been compared and evaluated using the performance metrics.(Here, RMSE and MAE are used).  

After that using real world data different from the original dataset, the  best model has been tested to get an estimate of its accuracy. After that, the same steps has been performed on rest of the dataset to create separate model for each sub-division(area) using the best approach.
For a better under standing of the methodology see the below given model architecture(fig. 7).  

## 6. Model Architecture 

![image.png](attachment:6e06f378-4a41-4a53-9756-9f34528dd5e2.png)


### About the Dataset :-
- The dataset(fig. 7) used in this system contains the rainfall of several regions in and across the country. It contains rainfall from 1901 – 2017 for the same. 
- There are in total 4188 rows present in the dataset. 
- Data has 36 sub divisions and 19 attributes (individual months, annual, combinations of 2-4 consecutive months).
- The dataset is been collected from Open Government Data (OGD) Platform India.
- All the attributes has the sum of amount of rainfall in mm.  

![image.png](attachment:bd77ede7-32ea-4bad-971c-7c95e616084e.png)  

![image.png](attachment:d68f5423-9c25-4ddc-958a-b2bb0c150470.png)  

![image.png](attachment:d4450017-399c-486c-b129-446813a7e74e.png)  

![image.png](attachment:de6079f9-03b9-46f9-868d-d229214819ef.png)  

![image.png](attachment:086c0ce9-2eea-4ee9-8825-1221c7e7b5c2.png)  


## 7. Implementation
The implementation of the project is divided into Seven steps :-  

### Step-1: Import Libraries:
Numpy, Pandas, Seaborn, Matplotlib, libraries for evaluating the dataset have been imported(fig. 13).   
Numpy is an open-source module that provides fast mathematical computation on arrays and matrices. Arrays are an integral part of the Machine Learning Ecosystem.  Pandas will be useful for performing operations on the data frame.   
Seaborn and Matplotlib are visualization tools that helps to visualize data in a better way.  
The required Deep Learning model(Sequential) and NN layers(Dense, Activation, Dropout, Bidirectional) and RNN Layer(LSTM) for model creation as well as scaling the data and error evaluation of the model have been imported.  
Xgboost regressor model and Elastic Net linear regression model have also been imported.  
The MinMaxScaler is used to scale the rainfall data into the range 0 to 1.  The data is trained after it is split into train set and test set.  

![image.png](attachment:1360895c-f8b6-4cb2-ad9b-0fc341a56380.png)  


### Step-2: Visualization of Data and Observing it to Learn form it.
Understanding the Data is important. Data visualization is a powerful technique that helps to know about the trends, patterns that our data follows. There are different techniques to visualize data, one such method is a correlation. Correlation tells us how one or more are related. If two variables are correlated, then we can tell that both are strongly dependent on each other. The variables that are strongly correlated to the target variable, are said to have more influence on the target variable.  

Correlation also helps us to remove certain values, as this is one of the feature extraction techniques. If two independent variables are strongly correlated with each other, we can remove any one of the variables. This may not cause any disruption to the dataset.  

Correlation can also be visualized using Heatmap. Heatmap is one of the visualizing graphs like Histograms, Boxplots that help us to know our data easily. As human minds are so complex to understand data from numbers, they can easily understand using pictures.  A histogram is used to summarize discrete or continuous data. In other words, it provides a visual interpretation. of numerical data by showing the number of data points that fall within a specified range of values.  

The data is imported(fig. 14- importing the dataset), viewed(fig. 15 - viewing the dataset) and using various visualization techniques it is evaluated.  

![image.png](attachment:0ea01d43-6896-45a1-819b-10e5963019c0.png)  

![image.png](attachment:bae065a3-018e-4063-869c-a9c7f35bd174.png)  

The different Plots as well as various observations are given below :-  

![image.png](attachment:1b1a5682-d0f1-45bb-a4c2-5e548c08a36c.png)  

The above figure(fig. 16 - line graph) :-
- Shows distribution of annual rainfall over years.
- Observed high amount of rainfall in 1950s.
- Before the 1950s there is an increasing trend while after 1960s there is a decreasing trend.  

![image.png](attachment:5554dce8-4774-48e7-b9c2-91112f092c13.png)  

![image.png](attachment:f17d960e-aaf3-4290-93f5-3800a76d3be3.png)  

The above figure(fig. 17 - histogram) :-
- Above histograms show the distribution of rainfall over months.
- Observed increase in amount of rainfall over months July, August, September.  

![image.png](attachment:dcf46140-0784-4251-8916-36692e8833e7.png)  

The above figure(fig. 18) :-
- Show the distribution of monthly rainfall from 1901 - 2017.
- The graphs shows that in month July India receives highest rainfall followed by months of August, June and September.
- Months of December , January and February receive the least amount of rainfall.  


![image.png](attachment:cbf5f4db-133f-460d-a184-f7cfa1ee257d.png)  

The above figure(fig. 19) :-
-  Show the distribution of rainfall over periods of Jan-Feb, Mar-Apr-May, Jun-Jul-Aug-Sep and Oct-Nov-Dec.
- The graphs clearly shows that amount of rainfall in high in the months June, July, Aug, Sept which is monsoon season in India.  

![image.png](attachment:fbc14a3d-9f7f-45be-8eac-8d81d6b763d8.png)  

The above figure(fig. 20) :-
- Shows Heat-Map with the co-relation(dependency) between the amounts of rainfall over months and annual rainfall.
- From above it is clear that if amount of rainfall is high in the months of July, August, September then the amount of rainfall will be high annually.
- It is also observed that if amount of rainfall in good in the months of October, November, December then the rainfall is going to be good in the overall year.  

![image.png](attachment:136c0279-d656-4048-9bfd-7fc5083c61ee.png)  

The above figure(fig. 16) :-
- Shows Heatmap, from which it is deduced that there are some rows where there is presence of null or Nan values.  



### Step-3: Prepare Dataset
For Prediction model preparation, only one sub-division has been chosen, here Jharkhand.  

The dataset is prepared for model creation by melting the columns containing the Monthly rainfall into rows(fig. 22) and then rearranging the values in proper order. The name of months have been substituted with corresponding number.  

Then again the data is sorted so that it remains in a Time series of monthly  rainfall From Jan 1901 - Dec 2017.  

![image.png](attachment:aedb67f1-a6d3-4821-956e-bac42b4fcc25.png)  

The dataset is grouped according to sub-division column and only the  months and year columns have been selected (fig. 23). 

![image.png](attachment:6eac36c1-638b-4b7e-adc5-79dfef8ecef2.png)  

![image.png](attachment:03d9d90c-3111-41c1-9de0-e2b6e0667c6a.png)  

![image.png](attachment:ca2bfd80-49c7-4a5a-bdea-5c0fc5a7f9fb.png)  

After melting the Year column(fig. 18), the data is sorted according to Year and index(fig. 24). Then, after renaming the columns, the month names are substituted with numbers(fig. 25).  

![image.png](attachment:8df2000c-dbdb-42e4-9a46-b16ecd73b1b1.png)  

After encoding the month names with numbers, the dataset has become a time-series dataset, with only rainfall amounts. Only the avg_rainfall column is selected and rest is discarded(fig. 26).  
This is the final time series dataset. Now we only need to do is data preprocessing.  

![image.png](attachment:37f57684-ef7f-406a-9b98-a139e529e978.png)  

Before data pre-processing, the time-series data is visualized to see the data(fig. 27).  



### Step-4: Data Preprocessing  
Data Preprocessing is the most vital step while preparing our dataset for model training.Data is often inconsistent, incomplete, and consists of noise or unwanted data. So, preprocessing is required. It involves certain steps like handling missing values, handling outliers, encoding techniques, scaling.  

Removing null values is most important because the presence of null values will disturb the  distribution of data, and may lead to false predictions. There is very less percent of null  values in the dataset.  

Missing values:  
Imputation is used for replacing missing values. There are few kinds of imputation techniques  like Mean imputation, Median imputation, Mode Imputation, Random Sampling imputation, etc. Based on the type of data we have, we can use the required imputation. We have used median  imputation to handle missing values.  

Handling Outliers:  
Outliers are nothing but an extreme value that deviates from the other observations in the dataset. These outliers are either removed or replaced with their nearest boundary value, either upper boundary or lower boundary value.  

Scaling: 
It is performed during the data preprocessing to handle highly varying magnitudes or values or units. If feature scaling is not done, then a machine learning algorithm tends to weigh greater values, higher and consider smaller values as the lower values, regardless of the unit of the values. Here, we are performing a min-max scaling.  

Min-max scale:
Also known as min-max scaling or min-max normalization, rescaling is the simplest method and consists in rescaling the range of features to scale the range in [0, 1] or [−1, 1]. Selecting the target range depends on the nature of the data.  

![image.png](attachment:c9b0e14b-a5da-42af-9af8-4ff63e597ed1.png)  

As  the Heatmap indicated there are some missing values in the dataset, so Mean value is used to fill them up(fig. 28).  

![image.png](attachment:bcb4dac9-e6da-422b-91b7-d79310918974.png)  

After the missing values are filled. the data is scaled to a standard, this is called standardization. Here(fig. 24). MinMaxScaler is being used for Standardization.  


### Step-5: Splitting the Dataset
Dividing the dataset into two sets should be done precisely. The dataset can be divided into the ratio of 80% train set, 20% test set or 70% train set, 30% test set, or any other way. The division of the dataset also affects the accuracy of the training model. A slicing operation can be performed to separate the dataset.  

It is to be noted that while splitting the dataset, assuring that the test set holds an equivalent features as the train set is necessary. The datasets must also be statistically meaningful.  

The dataset is divided into :- 
- 80% training dataset
- 20% testing dataset  

For prediction data have been formatted a in the way, given the rainfall in the last one month we try to predict the rainfall in the next consecutive month. This is called one-step forecasting.  
Predicting one time steps into the future is called one-step time series forecasting.  

![image.png](attachment:874da0cd-08d4-4f27-af83-caf567e935c2.png)  

After data pre-processing, the data needs to be divided into Training and Test dataset. In the above figure(fig. 30) . The data is divided into Train and Test sets for both ML and DL models. For Deep Learning model, the scaled data have been used to create model, while the Machine Learning model uses the original dataset. So, 4 sets of data have been created.  

![image.png](attachment:08a13d24-cc10-41f4-b13c-d0449046af32.png)  

In the above figure(fig. 31), the create_dataset function is used to arrange the datasets into one-step forecasting form.  

### Step-6: Model Training
For Training the model, 2 RNN (Recurrent Neural Networks), one with 2 stacked LSTM layer with 4 Dense Layers with alternative Dropout layers in between. And another with Bi-directional LSTM have been created.  2 ML model, Xgboost and Elastic Net Regression models have also been created. The accuracy is then Measured in RMSE(Root Mean Squared Error) and Mean Absolute Error(MAE).  

![image.png](attachment:48e2a06c-7bc0-41e0-9265-dde141ddf1ef.png)  

The above figure(fig. 32) shows The Stacked LSTM model.  

![image.png](attachment:9b5fbe96-af99-4404-9716-a1add96e139e.png)  

The above figure(fig. 33) shows The Stacked bidirectional LSTM model.  

![image.png](attachment:4c6ef45a-06c0-45b3-8e1b-001193258cf3.png)  

The above figure(fig. 29) shows The ML models (Elastic Net and Xgboost respectively).  

![image.png](attachment:c8398d18-b4d2-42c1-8acb-26a6a9dd4baf.png)  

The above figure(fig. 35) shows model creation through 4 approaches.  




### Step 7: Model Selection
After creating model, the performance of each model have been checked. Mean absolute Error and Root Mean Squared Error have been used as Performance metrics. Then, the data have been viewed by plotting the predicted values over true values for visual comparison.  

![image.png](attachment:f60a7fb7-4ae7-42f4-96d6-46672dc439c5.png)  

The above figure(fig. 36) shows performance metrics of Elastic Net model.
- Model :-- Elastic Net
  - Training data score: 100.99 RMSE
  - Test data score: 95.26 RMSE
  - Training data score: 72.19 MAE 
  - Test data score: 69.09 MAE  

![image.png](attachment:84d5a8f3-c207-43b7-b25d-4ca3199a80a8.png)

The above figure(fig. 37) shows performance metrics of Xgboost model.
- Model :-- XgBoost
  - Training data score: 31.96 RMSE
  - Test data score: 134.08 RMSE
  - Training data score: 15.61 MAE 
  - Test data score: 88.06 MAE  

![image.png](attachment:aa8a96a7-612b-4046-a946-9ecf7e3bd9c1.png)  

The above figure(fig. 38) shows performance metrics of LSTM Bi-Directional model.
- Model :-- LSTM Bidirectional
  - Training data score: 100.96 RMSE
  - Test data score: 95.44 RMSE
  - Training data score: 72.67 MAE 
  - Test data score: 69.22 MAE  

![image.png](attachment:b2a86afc-7997-433f-be35-e52a7e901960.png)  

The above figure(fig. 39) shows performance metrics of LSTM Stacked model.
- Model :-- LSTM Stacked
  - Training data score: 100.47 RMSE
  - Test data score: 97.49 RMSE
  - Training data score: 88.43 MAE 
  - Test data score: 77.66 MAE  


From the above plots, It is inferred infer that Elastic net and both LSTM models behave similarly and have given stable results. The Xgboost model while does better than all model when predicting the training dataset, it show instability when predicting the test dataset.  

The RSME and MAE data of each model have been plotted on a bar graph(fig. 40). The conclusion drawn is that the Elastic net training model preforms the best abet by a small margin than LSTM models. Xgboost shows the problem of Overfitting in its prediction set while both LSTM models performs almost same with LSTM Bi-Directional model being better of the 2 by a small margin.  
Now that, the best training model for the collected data have been found. Elastic Net would be used to create Models for other sub-divisions.  

![image.png](attachment:e944d54b-dada-49f5-9dbe-ccbaa7633d34.png)  



## 8. Sample Output



The above output models(fig. 41) can then be uploaded on Github or any other repository and  any body can download and use them as they seem fit.
The models can easily be loaded(fig. 42 & 43) and then used to predict rainfall:-  

![image.png](attachment:a4a417aa-62b5-4cf2-b28c-ad2a7ce989f9.png)  

![image.png](attachment:2dfa8a2c-b81c-4bc2-8881-8e1cfba90c99.png)  



## 10. Conclusion & Future Scope
Rainfall being one of the sole responsibilities for maximum economy of India, it should be considered the primary concern for most of us. Currently, rainfall prediction has become one of the key factors for most of the water conservation systems in and across country. One of the biggest challenges is the complexity present in rainfall data. Most of the rainfall prediction system, nowadays are unable to find the hidden features or any non-linear patterns present in the data due to use of old statistical methods.  

Machine Learning approach can been used to tackle the issue. The Dataset of 4116 rows has used that is converted to month-wise time-series data of rainfall in various regions in and around the country. A small part of the data was considered for test run i.e of the various regions(sub-divisions)only one is used. The considered subset is divided into two parts i.e. train data and test data(80:20 split). Train data is used for training the algorithm and test data is for doing the verification and evaluation. 4 models - 2 Machine Learning models (Elastic Net & XGBoost) and 2 Deep Learning Neural Network models(Stacked LSTM & BiLSTM) have been configured to create the predictor models. All models are then compared based on their performance metrics(RMSE and MAE). The one with better performance metrics is considered and implemented on the rest of the data to create models for each region. We concluded that Elastic Net regression methods was better than other approaches and models for each sub-division is created.  

Currently the model can predict the rainfall for next month using the rainfall data of current month. This can further be improved in future so that prediction can be made for a each day or may be for an year in advance. The constraint to such improvement is basically the dataset. The current dataset is only provides us with monthly data only.   

A better data-set with daily entries if produced can enhance the model creation to be able to predict the amount to rainfall for each consecutive day. Similarly, the model could be enhanced to be able to predict rainfall amount a year in advance, by using Multistep-ahead forecasting methods, in a way that inputting the monthly rainfall data of each month of current year will predict the rainfall for the months of the next year.   


## 11. Bibliography

1. Praveen, B., Talukdar, S., Shahfahad et al. Analyzing trend and forecasting of rainfall changes in India using non-parametrical and machine learning approaches. Sci Rep 10, 10342 (2020). https://doi.org/10.1038/s41598-020-67228-7
2. Demeke Endalie, Getamesay Haile, Wondmagegn Taye; Deep learning model for daily rainfall prediction: case study of Jimma, Ethiopia. Water Supply 1 March 2022; 22 (3): 3448–3461. doi: https://doi.org/10.2166/ws.2021.391
3. B.Meena Preethi, R. Gowtham, S.Aishvarya, S.Karthick,  D.G.Sabareesh,.ainfall Prediction using Machine Learning and Deep Learning Algorithms. DOI link: https://doi.org/10.35940/IJRTE.D6611.1110421
4. Machine Learning Geeks Of Geeks . Link:- https://www.geeksforgeeks.org/machine-learning/

