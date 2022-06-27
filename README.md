# major_project
![](vertopal_a73e3975d70747a9bedc0205a102929e/media/image2.png){width="6.997221128608924in"
height="6.997221128608924in"}![](vertopal_a73e3975d70747a9bedc0205a102929e/media/image3.png){width="1.3583333333333334in"
height="1.4149300087489063in"}

+----------------------------------------------------------------------+
| **Rainfall prediction model using ML**                               |
|                                                                      |
| A major project report submitted in partial fulfillment of the       |
|                                                                      |
| requirements for                                                     |
|                                                                      |
| the award of the Degree of Bachelor of Technology                    |
|                                                                      |
| In                                                                   |
|                                                                      |
| COMPUTER SCIENCE & ENGINEERING                                       |
|                                                                      |
| Submitted by                                                         |
|                                                                      |
| Abhijeet Paul(Regd. No. 1801292002)                                  |
|                                                                      |
| Devashish Kumar Pan(Regd. No. 1801292054)                            |
|                                                                      |
| Somit Shaw(Regd. No. 1801292160)                                     |
|                                                                      |
| Under the Guidance of                                                |
|                                                                      |
| Prof. P. Sharada                                                     |
|                                                                      |
| GEC, Bhubaneswar                                                     |
|                                                                      |
| Department of Computer Science & Engineering                         |
|                                                                      |
| Gandhi Engineering College, Bhubaneswar                              |
|                                                                      |
| ![](vertopal_a73e3975d7                                              |
| 0747a9bedc0205a102929e/media/image1.png){width="6.687498906386701in" |
| height="0.9111111111111111in"}                                       |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| **Certificate**                                                      |
|                                                                      |
| > This is to certify that the Project entitled Rainfall Prediction   |
| > Model, being                                                       |
| >                                                                    |
| > submitted by Abhijeet Paul, Devashish Kumar Pan and Somit Shaw,    |
| > bearing                                                            |
| >                                                                    |
| > University Reg. No 1801292002, 1801292054 and 1801292160           |
| > respectively, for                                                  |
| >                                                                    |
| > the award of degree of Bachelor of Technology in Computer Science  |
| > &                                                                  |
| >                                                                    |
| > Engineering, is a record of bonafide Project work carried out by   |
| > him/her under                                                      |
| >                                                                    |
| > my supervision. The matter embodied in this project have not been  |
| > submitted for                                                      |
| >                                                                    |
| > the award of any other degree anywhere.                            |
|                                                                      |
| +--------------------------------+----------------------------+      |
| | > \_\_\_\_\_\_\_\_\_\_\_\_\_\_ | \_\_\_\_\_\_\_\_\_\_\_\_\_ |      |
| +================================+============================+      |
| | > HOD, CSE                     | Project Guide              |      |
| +--------------------------------+----------------------------+      |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| **Acknowledgement**                                                  |
|                                                                      |
| Our sincere thanks to Mohini P. Mishra, Professor and Head of        |
| Department of                                                        |
|                                                                      |
| Computer Science Engineering, Gandhi Engineering College,            |
| Bhubaneswar for his                                                  |
|                                                                      |
| encouragement and valuable suggestions during the period of our      |
| Project work.                                                        |
|                                                                      |
| No words would suffice to express my regards and gratitude to Prof.  |
| P. Sharada,                                                          |
|                                                                      |
| Department of Computer Science & Engineering, for his/her inspiring  |
| guidance and                                                         |
|                                                                      |
| > constant encouragement, immense support and help during the        |
| > project work.                                                      |
|                                                                      |
| +-----------+-------------------------------------+                  |
| | > Date:-  | Abhijeet Paul(Regd. No. 1801292002) |                  |
| +===========+=====================================+                  |
| | > Place:- | Devashish Kumar Pan(Regd. No.       |                  |
| +-----------+-------------------------------------+                  |
| |           | 1801292054)                         |                  |
| +-----------+-------------------------------------+                  |
|                                                                      |
| Somit Shaw(Regd. No. 1801292160)                                     |
+----------------------------------------------------------------------+

+------------------------------------------------------------------------+
| **Content**                                                            |
|                                                                        |
| +-------+-------------------------------------------------+----------+ |
| | Sl.No | Description                                     | > Pg.No. | |
| +=======+=================================================+==========+ |
| | 1\.   | > Abstract                                      | 1        | |
| +-------+-------------------------------------------------+----------+ |
| | 2\.   | > Introduction to the Rainfall Prediction Model | 2        | |
| +-------+-------------------------------------------------+----------+ |
| | 3     | > System Requirement and Specification          | 3        | |
| +-------+-------------------------------------------------+----------+ |
| | 4\.   | > Literature                                    | 4-11     | |
| +-------+-------------------------------------------------+----------+ |
| | 5\.   | > Methodology                                   | 12       | |
| +-------+-------------------------------------------------+----------+ |
| | 6\.   | > Model Architecture                            | > 13-16  | |
| +-------+-------------------------------------------------+----------+ |
| | 7\.   | > Implementation                                | > 17-42  | |
| +-------+-------------------------------------------------+----------+ |
| | 8\.   | > Sample Code                                   | > 43-57  | |
| +-------+-------------------------------------------------+----------+ |
| | 9\.   | > Sample Output                                 | > 58-59  | |
| +-------+-------------------------------------------------+----------+ |
| | 10\.  | > Conclusion & Future Scope                     | > 60-61  | |
| +-------+-------------------------------------------------+----------+ |
| | 11\.  | > Bibliography                                  | 62       | |
| +-------+-------------------------------------------------+----------+ |
+------------------------------------------------------------------------+

  -- -- --
        
        
        
        
        
        
        
        
        
        
        
  -- -- --

+----------------------------------------------------------------------+
| **1. Abstract**                                                      |
|                                                                      |
| > Rainfall has been a major concern these days. Weather conditions   |
| > have beenchanging for the last 10 years. This has paved the way    |
| > for drastic changes inpatterns of rainfall.                        |
| >                                                                    |
| > Rainfall prediction model mainly based on artificial neural        |
| > networks are not themainstream. This project does a comparative    |
| > study of four rainfall predictionapproaches and finds the best     |
| > one. The present technique to predict rainfalldoesn't work well    |
| > with the complex data present. The approaches which are beingused  |
| > now-a-days are statistical methods and numerical methods, which    |
| > don't workaccurately when there is any non-linear pattern.         |
| > Existing system fails wheneverthe complexity of the datasets which |
| > contains past rainfall data increases.Henceforth, to find the best |
| > way to predict rainfall, study of both Machine Learningand Deep    |
| > Learning methods must be performed and the approach which          |
| > givesbetter accuracy should be used in prediction.                 |
| >                                                                    |
| > Rainfall can be considered as the pillar for the primary source of |
| > most of theeconomy of our country, i.e. Agriculture. To do a       |
| > proper investment on agriculture,a proper estimation of rainfall   |
| > is needed. Irregular heavy rainfall may lead to thedestruction of  |
| > crops, heavy floods, that can cause harm to human life. It         |
| > isimportant to exactly determine the rainfall for effective use of |
| > water resources,crop productivity, and pre-planning of water       |
| > structures.                                                        |
| >                                                                    |
| > People in coastal areas are in high risk of heavy rainfall and     |
| > floods, so they shouldbe aware of the rainfall much earlier so     |
| > that they can plan their stay accordingly.For areas which have     |
| > less rainfall and faces water scarcity should have                 |
| > rainwaterharvesters, which can collect the rainwater. To establish |
| > a proper rainwaterharvester, rainfall estimation is required.      |
| > Weather forecasting is the easiest andfastest way to get a greater |
| > outreach. This research work can be used by all theweather         |
| > forecasting channels, so that the prediction news can be more      |
| > accurateand can spread to all parts of the country.                |
|                                                                      |
| 1                                                                    |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| **2. Introduction to the Rainfall Prediction Model**                 |
|                                                                      |
| > The objective of the Rainfall prediction model is basically to     |
| > predict theapproximate amount of rainfall an area will receive in  |
| > the next month using therainfall amount of current month as input. |
| > It also gives us a measure of the trendor pattern the rainfall     |
| > will follow. Which is very helpful for the farmers, as It willgive |
| > appropriate time for them to prepare for, in-case low rainfall is  |
| > predicted.                                                         |
| >                                                                    |
| > This will also help governmental bodies to keep track of areas     |
| > which may face lessrainfall and may need assistance in form of     |
| > better water supply, creation of handpumps, etc.                   |
| >                                                                    |
| > For this, the past rainfall data have been gathered and arranged   |
| > it in a time seriesdataset. The dataset is then used to create     |
| > Prediction models using 4 differentapproaches. This is done to     |
| > explore a range of algorithms that are used to createtime-series   |
| > forecasting models.                                                |
| >                                                                    |
| > While any of them could have been used to create the required      |
| > model, bestpractice is to test a few of the prominent ones over a  |
| > subset of the dataset andthen compare them to get a measure of     |
| > which algorithm is best of the dataset.And then use it to create   |
| > model for the whole dataset.                                       |
|                                                                      |
| 2                                                                    |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| **3. System Requirement and Specification**                          |
|                                                                      |
| > System requirements are the configuration that a system must have  |
| > in order for ahardware or software application to run smoothly and |
| > efficiently. Failure to meetthese requirements can result in       |
| > installation problems or performance problems.The former may       |
| > prevent a device or application from getting installed, whereasthe |
| > latter may cause a product to malfunction or perform below         |
| > expectation oreven to hang or crash. System requirements are also  |
| > known as minimum systemrequirements.                               |
| >                                                                    |
| > **The system must have following requirement:**                    |
| >                                                                    |
| > ➢ Central Processing Unit (CPU) -- Intel core i5 6thgeneration     |
| > processor or higher. An AMD equivalent processor will also be      |
| > optimal.                                                           |
|                                                                      |
| +---+-----------------------------------------------------------+    |
| | ➢ | ➢ Ram: 4GB minimum, 8 GB or higher is recommended.        |    |
| +===+===========================================================+    |
| | ➢ | > Hard Disk: 250 GB.                                      |    |
| +---+-----------------------------------------------------------+    |
| | ➢ | ➢ Graphics Processing Unit(GPU) -- NVIDIA GeForce GTX960  |    |
| |   | or higher. AMD GPUs                                       |    |
| +---+-----------------------------------------------------------+    |
|                                                                      |
| > are not able to perform deep perform regardless.                   |
| >                                                                    |
| > ➢ OS: Ubuntu or Microsoft Windows 10.I recommend updating Windows  |
| > 10 to the latest version before proceeding forward.                |
|                                                                      |
| +---+-----------------------------------------------------------+    |
| | ➢ | ➢ Programming: Python 3.6 or above version and related    |    |
| |   | library files.                                            |    |
| +===+===========================================================+    |
| | ➢ | > Environment: jupyter notebook                           |    |
| +---+-----------------------------------------------------------+    |
|                                                                      |
| 3                                                                    |
+----------------------------------------------------------------------+

![](vertopal_a73e3975d70747a9bedc0205a102929e/media/image5.png){width="3.4375in"
height="2.283653762029746in"}![](vertopal_a73e3975d70747a9bedc0205a102929e/media/image6.png){width="3.4361100174978128in"
height="2.2827307524059495in"}

+----------------------------------------------------------------------+
| **4. Literature**                                                    |
|                                                                      |
| > **Linear Regression**                                              |
| >                                                                    |
| > Linear Regression is a machine learning algorithm based on         |
| > supervised learning. Itperforms a regression task. Regression      |
| > models a target prediction value based onindependent variables. It |
| > is mostly used for finding out the relationship betweenvariables   |
| > and forecasting. Different regression models differ based on --    |
| > the kind ofrelationship between dependent and independent          |
| > variables they are considering,and the number of independent       |
| > variables getting used.                                            |
|                                                                      |
| fig. 1 - Linear Regression best fit line.                            |
|                                                                      |
| > Linear regression performs the task to predict a dependent         |
| > variable value (y)based on a given independent variable (x).       |
| >                                                                    |
| > So, this regression technique finds out a linear relationship      |
| > between x (input) andy(output). Hence, the name is Linear          |
| > Regression.                                                        |
| >                                                                    |
| > In the figure(fig. 1) above, X (input) is the work experience and  |
| > Y (output) is thesalary of a person. The regression line is the    |
| > best fit line for our model.                                       |
| >                                                                    |
| > Hypothesis function for Linear Regression :                        |
| >                                                                    |
| > ![](vertopal_a73e3975d70                                           |
| 747a9bedc0205a102929e/media/image4.png){width="1.6055555555555556in" |
| > height="0.3263888888888889in"}                                     |
|                                                                      |
| 4                                                                    |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > While training the model we are given :                            |
| >                                                                    |
| > x: input training data (univariate -- one input                    |
| > variable(parameter))                                               |
| >                                                                    |
| > y: labels to data (supervised learning)                            |
| >                                                                    |
| > When training the model -- it fits the best line to predict the    |
| > value of y for a givenvalue of x. The model gets the best          |
| > regression fit line by finding the best θ1 and θ2                  |
| >                                                                    |
| > values.                                                            |
| >                                                                    |
| > θ1 : intercept                                                     |
| >                                                                    |
| > θ2: coefficient of x                                               |
|                                                                      |
| Once we find the best θ1 and θ2 values, we get the best fit line. So |
| when we are                                                          |
|                                                                      |
| > finally using our model for prediction, it will predict the value  |
| > of y for the inputvalue of x.                                      |
| >                                                                    |
| > **How to update θ1 and θ2 values to get the best fit line ?**      |
| >                                                                    |
| > **Cost Function (J):**                                             |
| >                                                                    |
| > By achieving the best-fit regression line, the model aims to       |
| > predict y value suchthat the error difference between predicted    |
| > value and true value is minimum. So,it is very important to update |
| > the θ1 and θ2 values, to reach the best value that                 |
| >                                                                    |
| > minimize the error between predicted y value (pred) and true y     |
| > value (y).                                                         |
| >                                                                    |
| > ![](vertopal_a73e3975d70                                           |
| 747a9bedc0205a102929e/media/image7.png){width="2.4166666666666665in" |
| > height="0.6527777777777778in"}                                     |
| >                                                                    |
| > ![](vertopal_a73e3975d7                                            |
| 0747a9bedc0205a102929e/media/image8.png){width="2.198611111111111in" |
| > height="0.6625in"}                                                 |
| >                                                                    |
| > Cost function(J) of Linear Regression is the Root Mean Squared     |
| > Error (RMSE)between predicted y value (pred) and true y value (y). |
|                                                                      |
| 5                                                                    |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > **Gradient Descent**:                                              |
| >                                                                    |
| > Gradient Descent is an optimization algorithm used for minimizing  |
| > the costfunction in various machine learning algorithms. It is     |
| > basically used for updatingthe parameters of the learning model.   |
| > To update θ1 and θ2 values in order to                             |
| >                                                                    |
| > reduce Cost function (minimizing RMSE value) and achieving the     |
| > best fit line themodel uses Gradient Descent. The idea is to start |
| > with random θ1 and θ2 values                                       |
| >                                                                    |
| > and then iteratively updating the values, reaching minimum cost.   |
|                                                                      |
| +-------------------+---+-------------------+-------------------+    |
| | > ![](ver         | ➢ | θj                | > : Weights of    |    |
| | topal_a73e3975d70 |   |                   | > the hypothesis. |    |
| | 747a9bedc0205a102 |   |                   |                   |    |
| | 929e/media/image9 |   |                   |                   |    |
| | .png){width="3.43 |   |                   |                   |    |
| | 19444444444445in" |   |                   |                   |    |
| | > height="5.28    |   |                   |                   |    |
| | 4722222222222in"} |   |                   |                   |    |
| +===================+===+===================+===================+    |
| |                   | ➢ | hθ(xi) :          | > for ith         |    |
| |                   |   | predicted y value |                   |    |
| +-------------------+---+-------------------+-------------------+    |
| |                   | ➢ | > input.          |                   |    |
| +-------------------+---+-------------------+-------------------+    |
| |                   |   | > j               | > : Feature index |    |
| |                   |   |                   | > number (can be  |    |
| +-------------------+---+-------------------+-------------------+    |
| |                   | ➢ | > 0, 1, 2,        |                   |    |
| |                   |   | > \...\..., n).   |                   |    |
| +-------------------+---+-------------------+-------------------+    |
| |                   |   | > α               | > : Learning Rate |    |
| |                   |   |                   | > of Gradient     |    |
| +-------------------+---+-------------------+-------------------+    |
| |                   |   | > Descent.        |                   |    |
| +-------------------+---+-------------------+-------------------+    |
|                                                                      |
| > fig. 2 - Gradient Decent Formulas                                  |
|                                                                      |
| 6                                                                    |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > **Regularization in Machine Learning**                             |
| >                                                                    |
| > Overfitting(fig. 3) is a phenomenon that occurs when a Machine     |
| > Learning model isconstraint to training set and not able to        |
| > perform well on unseen data.                                       |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image10.png){width="5.916666666666667in" |
| height="2.4166666666666665in"}                                       |
|                                                                      |
| fig. 3 - Under-Fitting, Over-Fitting and Best-Fit                    |
|                                                                      |
| > Regularization works by adding a penalty or complexity term to the |
| > complex. Thecommonly used regularization techniques are :          |
| >                                                                    |
| > 1\. L1 regularization(Lasso regression) :- It is calculated by     |
| > multiplying the sum ofabsolute weight of each individual feature   |
| > to lambda which is then added aspenalty to the cost function.      |
| >                                                                    |
| > 2\. L2 regularization(Ridge regression) :- It is calculated by     |
| > multiplying the sum ofsquared weight of each individual feature to |
| > lambda which is then added aspenalty to the cost function.         |
| >                                                                    |
| > Ridge regression is mostly used to reduce the overfitting in the   |
| > model, and itincludes all the features present in the model. It    |
| > reduces the complexity of themodel by shrinking the coefficients.  |
| >                                                                    |
| > Lasso regression helps to reduce the overfitting in the model as   |
| > well as featureselection                                           |
|                                                                      |
| 7                                                                    |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > **Elastic Net Regression**                                         |
| >                                                                    |
| > Elastic Net(fig. 4) is a regression method that performs variable  |
| > selection andregularization both simultaneously. Elastic net is a  |
| > popular type of regularizedlinear regression that combines two     |
| > popular penalties, specifically the L1 and L2penalty functions.    |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image11.png){width="5.430555555555555in" |
| height="5.3236100174978125in"}                                       |
|                                                                      |
| fig. 4 - Graphical representation of Elastic Net Regression.         |
|                                                                      |
| 8                                                                    |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > **Long short-term memory(LSTM)**                                   |
| >                                                                    |
| > Long short-term memory(fig. 5) (LSTM) is an artificial recurrent   |
| > neural network(RNN) used in the field of deep learning. A common   |
| > LSTM unit is composed of acell, an input gate, an output gate and  |
| > a forget gate. The cell remembers valuesover arbitrary time        |
| > intervals and the three gates regulate the flow of informationinto |
| > and out of the cell. The functions of the three types of gates     |
| > are:                                                               |
| >                                                                    |
| > ◆ forget gate -- controls how much information the memory cell     |
| > will receive from the memory cell from the previous step           |
| >                                                                    |
| > ◆ update (input) gate -- decides whether the memory cell will be   |
| > updated. Also, it controls how much information the current memory |
| > cell will receive from a potentially new memory cell.              |
| >                                                                    |
| > ◆ output gate -- controls the value of the next hidden state       |
|                                                                      |
| ![](vertopal                                                         |
| _a73e3975d70747a9bedc0205a102929e/media/image12.png){width="5.875in" |
| height="3.3916666666666666in"}                                       |
|                                                                      |
| fig. 5 - A single LSTM cell.                                         |
|                                                                      |
| > LSTM networks are used to classifying, processing and making       |
| > predictions basedon time series data. LSTM is a special type of    |
| > recurrent neural network.Specifically, this architecture is        |
| > introduced to solve the problem of vanishing and                   |
|                                                                      |
| 9                                                                    |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > exploding gradients. In addition, this type of network is better   |
| > for maintaininglong-range connections, recognizing the             |
| > relationship between values at thebeginning and end of a sequence. |
| >                                                                    |
| > **Stacked LSTM**                                                   |
| >                                                                    |
| > A Stacked LSTM architecture can be defined as an LSTM model        |
| > comprised ofmultiple LSTM layers. An LSTM layer above provides a   |
| > sequence output rather than asingle value output to the LSTM layer |
| > below. Specifically, one output per input timestep, rather than    |
| > one output time step for all input time steps.                     |
| >                                                                    |
| > **Bidirectional LSTM**                                             |
| >                                                                    |
| > A Bidirectional LSTM, or BiLSTM, is a sequence processing model    |
| > that consists oftwo LSTM layers: one taking the input in a forward |
| > direction, and the other in abackwards direction. BiLSTMs          |
| > effectively increase the amount of informationavailable to the     |
| > network, improving the context available to the algorithm.         |
| >                                                                    |
| > Unlike standard LSTM, the input flows in both directions, and it's |
| > capable ofutilizing information from both sides. It's also a       |
| > powerful tool for modeling thesequential dependencies between      |
| > inputs and outputs in both directions of thesequence.              |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image13.png){width="6.916666666666667in" |
| height="2.594443350831146in"}                                        |
|                                                                      |
| fig. 6 - Single BiLSTM layer.                                        |
|                                                                      |
| 10                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > In summary, BiLSTM adds one more LSTM layer, which reverses the    |
| > direction ofinformation flow. Briefly, it means that the input     |
| > sequence flows backward in theadditional LSTM layer. Then we       |
| > combine the outputs from both LSTM layers inseveral ways, such as  |
| > average, sum, multiplication, or concatenation.This typeof         |
| > architecture has many advantages in real-world problems. The main  |
| > reason isthat every component of an input sequence has information |
| > from both the pastand present. For this reason, BiLSTM can produce |
| > a more meaningful output,combining LSTM layers from both           |
| > directions.                                                        |
| >                                                                    |
| > **XGBoost**                                                        |
| >                                                                    |
| > XGBoost is a decision-tree-based ensemble Machine Learning         |
| > algorithm thatuses a gradient boosting framework. In prediction    |
| > problems involving small-to-medium structured/tabular data,        |
| > decision tree based algorithms are consideredbest-in-class right   |
| > now.                                                               |
| >                                                                    |
| > XGBoost is an implementation of gradient boosted decision trees    |
| > designed forspeed and performance. It is a powerful approach for   |
| > building supervisedregression models. The validity of this         |
| > statement can be inferred by knowingabout its (XGBoost) objective  |
| > function and base learners.                                        |
| >                                                                    |
| > The objective function contains loss function and a regularization |
| > term. It tellsabout the difference between actual values and       |
| > predicted values, i.e how far themodel results are from the real   |
| > values. The most common loss functions inXGBoost for regression    |
| > problems is reg:linear, and that for binary classificationis       |
| > reg:logistics.                                                     |
| >                                                                    |
| > Ensemble learning involves training and combining individual       |
| > models (known asbase learners) to get a single prediction, and     |
| > XGBoost is one of the ensemblelearning methods. XGBoost expects to |
| > have the base learners which areuniformly bad at the remainder so  |
| > that when all the predictions are combined,bad predictions cancels |
| > out and better one sums up to form final goodpredictions.          |
|                                                                      |
| 11                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| **5. Methodology**                                                   |
|                                                                      |
| > The proposed methodology of this project is rather simple. Time    |
| > series data hasbeen collected from trusted data sources. Then      |
| > using various visualizationtechniques the data has been evaluated  |
| > to check for its accuracy and to find thevarious trends present in |
| > the dataset. Thereafter, having confirmed the trends aswell as the |
| > dependencies of various features present in the dataset, the       |
| > processfor creation of model from the data has been selected. In   |
| > this project one-stepforecasting method has been used, which means |
| > given the rainfall in the lastmonth the model tries to predict the |
| > rainfall in the next month.                                        |
| >                                                                    |
| > As the dataset is too big it has been divided into various subsets |
| > according to sub-divisions and only one sub-division(area) data    |
| > has been selected. In the next stepdata has been converted into    |
| > the correct format to fit it into the 4 models(2machine learning   |
| > models and 2 deep learning models). The processed data hasbeen     |
| > separated into training set and testing set then 4 models has been |
| > trainedusing the training dataset. Then using the test dataset the |
| > 4 approaches used tocreate models has been compared and evaluated  |
| > using the performancemetrics.(Here, RMSE and MAE are used).        |
| >                                                                    |
| > After that using real world data different from the original       |
| > dataset, the best modelhas been tested to get an estimate of its   |
| > accuracy. After that, the same steps hasbeen performed on rest of  |
| > the dataset to create separate model for each sub-division(area)   |
| > using the best approach.                                           |
| >                                                                    |
| > For a better under standing of the methodology see the below given |
| > modelarchitecture(fig. 7):-                                        |
|                                                                      |
| 12                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| **6. Model Architecture**                                            |
|                                                                      |
| ![](vertop                                                           |
| al_a73e3975d70747a9bedc0205a102929e/media/image14.png){width="6.3in" |
| height="3.845832239720035in"}                                        |
|                                                                      |
| fig. 7 - Model Architecture                                          |
|                                                                      |
| 13                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > **About the Dataset** :-                                           |
| >                                                                    |
| > \- The dataset(fig. 7) used in this system contains the rainfall   |
| > of several regions inand across the country. It contains rainfall  |
| > from 1901 -- 2017 for the same.                                    |
| >                                                                    |
| > \- There are in total 4188 rows present in the dataset.            |
| >                                                                    |
| > \- Data has 36 sub divisions and 19 attributes (individual months, |
| > annual,combinations of 2-4 consecutive months).                    |
| >                                                                    |
| > \- The dataset is been collected from Open Government Data (OGD)   |
| > Platform India.                                                    |
| >                                                                    |
| > \- All the attributes has the sum of amount of rainfall in mm.     |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image15.png){width="6.504166666666666in" |
| height="2.8847222222222224in"}                                       |
|                                                                      |
| fig. 8 - Spread-Sheet View of Dataset.                               |
|                                                                      |
| 14                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image16.png){width="6.998611111111111in" |
| height="3.798611111111111in"}                                        |
|                                                                      |
| fig. 9 - Information about dataset                                   |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image17.png){width="6.995833333333334in" |
| height="2.0486100174978126in"}                                       |
|                                                                      |
| fig. 10 - Jupyter notebook View of Dataset.                          |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image18.png){width="6.997222222222222in" |
| height="0.8055555555555556in"}                                       |
|                                                                      |
| fig. 11 - The Column names of Dataset.                               |
|                                                                      |
| 15                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
|   ------------------------------------------                         |
| -------------------------------------------------------------------- |
|   ![](vertopal_a73e3975d70747a9bedc0205a10                           |
| 2929e/media/image19.png){width="7.0in" height="5.230555555555555in"} |
|   ------------------------------------------                         |
| -------------------------------------------------------------------- |
|                                                                      |
| fig. 12 - The Subdivisions in the Dataset.                           |
|                                                                      |
| 16                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| **7. Implementation**                                                |
|                                                                      |
| > The implementation of the project is divided into Seven steps :-   |
| >                                                                    |
| > **Step-1: Import Libraries:**                                      |
| >                                                                    |
| > Numpy, Pandas, Seaborn, Matplotlib, libraries for evaluating the   |
| > dataset havebeen imported(fig. 13).                                |
| >                                                                    |
| > Numpy is an open-source module that provides fast mathematical     |
| > computation onarrays and matrices. Arrays are an integral part of  |
| > the Machine LearningEcosystem. Pandas will be useful for           |
| > performing operations on the data frame.                           |
| >                                                                    |
| > Seaborn and Matplotlib are visualization tools that helps to       |
| > visualize data in abetter way.                                     |
| >                                                                    |
| > The required Deep Learning model(Sequential) and NN layers(Dense,  |
| > Activation,Dropout, Bidirectional) and RNN Layer(LSTM) for model   |
| > creation as well as scalingthe data and error evaluation of the    |
| > model have been imported.                                          |
| >                                                                    |
| > Xgboost regressor model and Elastic Net linear regression model    |
| > have also beenimported.                                            |
| >                                                                    |
| > The MinMaxScaler is used to scale the rainfall data into the range |
| > 0 to 1. The datais trained after it is split into train set and    |
| > test set.                                                          |
|                                                                      |
| 17                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image20.png){width="5.516666666666667in" |
| height="3.3097222222222222in"}                                       |
|                                                                      |
| fig. 13 - Importing Various Libraries                                |
|                                                                      |
| > **Step-2: Visualization of Data and Observing it to Learn form     |
| > it.**                                                              |
| >                                                                    |
| > Understanding the Data is important. Data visualization is a       |
| > powerful techniquethat helps to know about the trends, patterns    |
| > that our data follows. There aredifferent techniques to visualize  |
| > data, one such method is a correlation.Correlation tells us how    |
| > one or more are related. If two variables are correlated,then we   |
| > can tell that both are strongly dependent on each other. The       |
| > variablesthat are strongly correlated to the target variable, are  |
| > said to have more influenceon the target variable.                 |
| >                                                                    |
| > Correlation also helps us to remove certain values, as this is one |
| > of the featureextraction techniques. If two independent variables  |
| > are strongly correlated witheach other, we can remove any one of   |
| > the variables. This may not cause anydisruption to the dataset.    |
| >                                                                    |
| > Correlation can also be visualized using Heatmap. Heatmap is one   |
| > of thevisualizing graphs like Histograms, Boxplots that help us to |
| > know our data easily.As human minds are so complex to understand   |
| > data from numbers, they can                                        |
|                                                                      |
| 18                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > easily understand using pictures. A histogram is used to summarize |
| > discrete orcontinuous data. In other words, it provides a visual   |
| > interpretation. of numericaldata by showing the number of data     |
| > points that fall within a specified range ofvalues.                |
| >                                                                    |
| > The data is imported(fig. 14- importing the dataset), viewed(fig.  |
| > 15 - viewing thedataset) and using various visualization           |
| > techniques it is evaluated.                                        |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image21.png){width="6.329165573053368in" |
| height="1.55in"}                                                     |
|                                                                      |
| fig. 14 - importing the dataset                                      |
|                                                                      |
| ![](vertopal_a73e3975d707                                            |
| 47a9bedc0205a102929e/media/image22.png){width="6.9944444444444445in" |
| height="1.8416666666666666in"}                                       |
|                                                                      |
| fig. 15 - viewing the dataset                                        |
|                                                                      |
| 19                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > The different Plots as well as various observations are given      |
| > below :-                                                           |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image23.png){width="5.995832239720035in" |
| height="3.575in"}                                                    |
|                                                                      |
| fig. 16 - Year-wise Annual Rainfall                                  |
|                                                                      |
| > The above figure(fig. 16 - line graph) :-                          |
| >                                                                    |
| > \- Shows distribution of annual rainfall over years.               |
| >                                                                    |
| > \- Observed high amount of rainfall in 1950s.                      |
| >                                                                    |
| > \- Before the 1950s there is an increasing trend while after 1960s |
| > there is adecreasing trend.                                        |
|                                                                      |
| 20                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > ![](vertopal_a73e3975d70                                           |
| 747a9bedc0205a102929e/media/image24.png){width="4.813888888888889in" |
| > height="0.5277777777777778in"}                                     |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image25.png){width="6.016666666666667in" |
| height="5.8375in"}                                                   |
|                                                                      |
| fig. 17 - Histogram Showing Rainfall Distribution.                   |
|                                                                      |
| > The above figure(fig. 17 - histogram) :-                           |
| >                                                                    |
| > \- Above histograms show the distribution of rainfall over months. |
| >                                                                    |
| > \- Observed increase in amount of rainfall over months July,       |
| > August, September.                                                 |
|                                                                      |
| 21                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image26.png){width="5.858332239720035in" |
| height="3.6347222222222224in"}                                       |
|                                                                      |
| fig. 18 - Year -wise Rainfall of all different months.               |
|                                                                      |
| > The above figure(fig. 18) :-                                       |
| >                                                                    |
| > \- Show the distribution of monthly rainfall from 1901 - 2017.     |
| >                                                                    |
| > \- The graphs shows that in month July India receives highest      |
| > rainfall followed bymonths of August, June and September.          |
| >                                                                    |
| > \- Months of December , January and February receive the least     |
| > amount of rainfall.                                                |
|                                                                      |
| 22                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image27.png){width="5.930555555555555in" |
| height="3.8680555555555554in"}                                       |
|                                                                      |
| fig. 19 - Year-wise rainfall of various seasons.                     |
|                                                                      |
| > The above figure(fig. 19) :-                                       |
| >                                                                    |
| > \- Show the distribution of rainfall over periods of Jan-Feb,      |
| > Mar-Apr-May, Jun-Jul-Aug-Sep and Oct-Nov-Dec.                      |
| >                                                                    |
| > \- The graphs clearly shows that amount of rainfall in high in the |
| > months June, July,Aug, Sept which is monsoon season in India.      |
|                                                                      |
| 23                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image28.png){width="6.530555555555556in" |
| height="5.406944444444444in"}                                        |
|                                                                      |
| fig. 20 - Heatmap Showing Correlation between Various Columns.       |
|                                                                      |
| > The above figure(fig. 20) :-                                       |
| >                                                                    |
| > \- Shows Heat-Map with the co-relation(dependency) between the     |
| > amounts ofrainfall over months and annual rainfall.                |
| >                                                                    |
| > \- From above it is clear that if amount of rainfall is high in    |
| > the months of July,August, September then the amount of rainfall   |
| > will be high annually.                                             |
| >                                                                    |
| > \- It is also observed that if amount of rainfall in good in the   |
| > months of October,November, December then the rainfall is going to |
| > be good in the overall year.                                       |
|                                                                      |
| 24                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
|   ------------------------------------------                         |
| -------------------------------------------------------------------- |
|   ![](vertopal_a73e3975d70747a9bedc0205a10                           |
| 2929e/media/image29.png){width="7.0in" height="4.833333333333333in"} |
|   ------------------------------------------                         |
| -------------------------------------------------------------------- |
|                                                                      |
| fig. 21 - Heatmap Showing Presence of Null values.                   |
|                                                                      |
| > The above figure(fig. 16) :-                                       |
| >                                                                    |
| > \- Shows Heatmap, from which it is deduced that there are some     |
| > rows where thereis presence of null or Nan values.                 |
|                                                                      |
| 25                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > **Step-3: Prepare Dataset**                                        |
| >                                                                    |
| > For Prediction model preparation, only one sub-division has been   |
| > chosen, hereJharkhand.                                             |
| >                                                                    |
| > The dataset is prepared for model creation by melting the columns  |
| > containing theMonthly rainfall into rows(fig. 22) and then         |
| > rearranging the values in proper order.The name of months have     |
| > been substituted with corresponding number.                        |
| >                                                                    |
| > Then again the data is sorted so that it remains in a Time series  |
| > of monthlyrainfall From Jan 1901 - Dec 2017.                       |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image30.png){width="5.645833333333333in" |
| height="2.1569444444444446in"}                                       |
|                                                                      |
| fig. 22 - Selecting only one Subdivision.                            |
|                                                                      |
| > The dataset is grouped according to sub-division column and only   |
| > the months andyear columns have been selected (fig. 23).           |
|                                                                      |
| 26                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image31.png){width="3.808333333333333in" |
| height="2.9652777777777777in"}                                       |
|                                                                      |
| fig. 23 - Changing Months from columns to rows.                      |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image32.png){width="4.979166666666667in" |
| height="2.9791666666666665in"}                                       |
|                                                                      |
| fig. 24 - Sorting the rows according to year and then index to       |
| create the time series.                                              |
|                                                                      |
| 27                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image33.png){width="4.759722222222222in" |
| height="3.2597222222222224in"}                                       |
|                                                                      |
| fig. 25 - Mapping the months values with numbers.                    |
|                                                                      |
| > After melting the Year column(fig. 18), the data is sorted         |
| > according to Year andindex(fig. 24). Then, after renaming the      |
| > columns, the month names are substitutedwith numbers(fig. 25).     |
|                                                                      |
| 28                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image34.png){width="4.280555555555556in" |
| height="4.073611111111111in"}                                        |
|                                                                      |
| fig. 26 - Selecting only the Avg_rainfall column.                    |
|                                                                      |
| > After encoding the month names with numbers, the dataset has       |
| > become a time-series dataset, with only rainfall amounts. Only the |
| > avg_rainfall column is selectedand rest is discarded(fig. 26).     |
| >                                                                    |
| > This is the final time series dataset. Now we only need to do is   |
| > data preprocessing.                                                |
|                                                                      |
| 29                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image35.png){width="6.997222222222222in" |
| height="4.345833333333333in"}                                        |
|                                                                      |
| fig. 27 - Line plot view of Rainfall time series.                    |
|                                                                      |
| > Before data pre-processing, the time-series data is visualized to  |
| > see the data(fig.27).                                              |
|                                                                      |
| 30                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > **Step-4: Data Preprocessing**                                     |
| >                                                                    |
| > Data Preprocessing is the most vital step while preparing our      |
| > dataset for modeltraining.Data is often inconsistent, incomplete,  |
| > and consists of noise or unwanteddata. So, preprocessing is        |
| > required. It involves certain steps like handling missingvalues,   |
| > handling outliers, encoding techniques, scaling.                   |
| >                                                                    |
| > Removing null values is most important because the presence of     |
| > null values willdisturb the distribution of data, and may lead to  |
| > false predictions. There is veryless percent of null values in the |
| > dataset.                                                           |
| >                                                                    |
| > Missing values:                                                    |
| >                                                                    |
| > Imputation is used for replacing missing values. There are few     |
| > kinds of imputationtechniques like Mean imputation, Median         |
| > imputation, Mode Imputation, RandomSampling imputation, etc. Based |
| > on the type of data we have, we can use therequired imputation. We |
| > have used median imputation to handle missing values.              |
| >                                                                    |
| > Handling Outliers:                                                 |
| >                                                                    |
| > Outliers are nothing but an extreme value that deviates from the   |
| > otherobservations in the dataset. These outliers are either        |
| > removed or replaced withtheir nearest boundary value, either upper |
| > boundary or lower boundary value.                                  |
| >                                                                    |
| > Scaling:                                                           |
| >                                                                    |
| > It is performed during the data preprocessing to handle highly     |
| > varying magnitudesor values or units. If feature scaling is not    |
| > done, then a machine learning algorithmtends to weigh greater      |
| > values, higher and consider smaller values as the lowervalues,     |
| > regardless of the unit of the values. Here, we are performing a    |
| > min-maxscaling.                                                    |
|                                                                      |
| 31                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > Min-max scale:                                                     |
| >                                                                    |
| > Also known as min-max scaling or min-max normalization, rescaling  |
| > is the simplestmethod and consists in rescaling the range of       |
| > features to scale the range in \[0, 1\]or \[−1, 1\]. Selecting the |
| > target range depends on the nature of the data.                    |
|                                                                      |
| ![](vertopal_                                                        |
| a73e3975d70747a9bedc0205a102929e/media/image36.png){width="4.8125in" |
| height="2.5930555555555554in"}                                       |
|                                                                      |
| fig. 28 - Selecting dataset values as float32.                       |
|                                                                      |
| > As the Heatmap indicated there are some missing values in the      |
| > dataset, so Meanvalue is used to fill them up(fig. 28).            |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image37.png){width="5.229166666666667in" |
| height="2.0in"}                                                      |
|                                                                      |
| fig. 29 - Scaling the dataset.                                       |
|                                                                      |
| > After the missing values are filled. the data is scaled to a       |
| > standard, this is calledstandardization. Here(fig. 24).            |
| > MinMaxScaler is being used for Standardization.                    |
|                                                                      |
| 32                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > **Step-5: Splitting the Dataset**                                  |
| >                                                                    |
| > Dividing the dataset into two sets should be done precisely. The   |
| > dataset can bedivided into the ratio of 80% train set, 20% test    |
| > set or 70% train set, 30% test set,or any other way. The division  |
| > of the dataset also affects the accuracy of thetraining model. A   |
| > slicing operation can be performed to separate the dataset.        |
| >                                                                    |
| > It is to be noted that while splitting the dataset, assuring that  |
| > the test set holds anequivalent features as the train set is       |
| > necessary. The datasets must also bestatistically meaningful.      |
| >                                                                    |
| > The dataset is divided into :-                                     |
| >                                                                    |
| > \- 80% training dataset                                            |
| >                                                                    |
| > \- 20% testing dataset                                             |
| >                                                                    |
| > For prediction data have been formatted a in the way, given the    |
| > rainfall in the lastone month we try to predict the rainfall in    |
| > the next consecutive month. This iscalled one-step forecasting.    |
| >                                                                    |
| > Predicting one time steps into the future is called one-step time  |
| > series forecasting.                                                |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image38.png){width="6.998611111111111in" |
| height="2.220832239720035in"}                                        |
|                                                                      |
| fig. 30 - Splitting the Dataset                                      |
|                                                                      |
| > After data pre-processing, the data needs to be divided into       |
| > Training and Testdataset. In the above figure(fig. 30) . The data  |
| > is divided into Train and Test sets                                |
|                                                                      |
| 33                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > for both ML and DL models. For Deep Learning model, the scaled     |
| > data have beenused to create model, while the Machine Learning     |
| > model uses the original dataset.So, 4 sets of data have been       |
| > created.                                                           |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image39.png){width="6.995833333333334in" |
| height="3.738888888888889in"}                                        |
|                                                                      |
| fig. 31 - Arranging the Input and output values for one-step         |
| forecasting.                                                         |
|                                                                      |
| > In the above figure(fig. 31), the create_dataset function is used  |
| > to arrange thedatasets into one-step forecasting form.             |
|                                                                      |
| 34                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > **Step-6: Model Training**                                         |
| >                                                                    |
| > For Training the model, 2 RNN (Recurrent Neural Networks), one     |
| > with 2 stackedLSTM layer with 4 Dense Layers with alternative      |
| > Dropout layers in between. Andanother with Bi-directional LSTM     |
| > have been created. 2 ML model, Xgboost andElastic Net Regression   |
| > models have also been created. The accuracy is thenMeasured in     |
| > RMSE(Root Mean Squared Error) and Mean Absolute Error(MAE).        |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image40.png){width="5.583333333333333in" |
| height="5.551388888888889in"}                                        |
|                                                                      |
| fig. 32 - Stacked LSTM model.                                        |
|                                                                      |
| > The above figure(fig. 32) shows The Stacked LSTM model.            |
|                                                                      |
| 35                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image41.png){width="5.563888888888889in" |
| height="5.541666666666667in"}                                        |
|                                                                      |
| fig. 33 - BiLSTM model.                                              |
|                                                                      |
| > The above figure(fig. 33) shows The Stacked bidirectional LSTM     |
| > model.                                                             |
|                                                                      |
| 36                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image42.png){width="5.958333333333333in" |
| height="1.3847222222222222in"}                                       |
|                                                                      |
| fig. 34 - Elastic Net & XGBoost models.                              |
|                                                                      |
| > The above figure(fig. 29) shows The ML models (Elastic Net and     |
| > Xgboostrespectively).                                              |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image43.png){width="6.093055555555556in" |
| height="3.595832239720035in"}                                        |
|                                                                      |
| fig. 35 - Fitting the models.                                        |
|                                                                      |
| > The above figure(fig. 35) shows model creation through 4           |
| > approaches.                                                        |
|                                                                      |
| 37                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > **Step 7: Model Selection**                                        |
| >                                                                    |
| > After creating model, the performance of each model have been      |
| > checked. Meanabsolute Error and Root Mean Squared Error have been  |
| > used as Performancemetrics. Then, the data have been viewed by     |
| > plotting the predicted values overtrue values for visual           |
| > comparison.                                                        |
|                                                                      |
| ![](vertopal_a73e3975d707                                            |
| 47a9bedc0205a102929e/media/image44.png){width="5.9944444444444445in" |
| height="3.8208333333333333in"}                                       |
|                                                                      |
| fig. 36 - Comparison of Original and Predicted Time Series(Elastic   |
| Net model)                                                           |
|                                                                      |
| > The above figure(fig. 36) shows performance metrics of Elastic Net |
| > model.                                                             |
| >                                                                    |
| > Model :\-- Elastic Net                                             |
| >                                                                    |
| > Training data score: 100.99 RMSE                                   |
| >                                                                    |
| > Test data score: 95.26 RMSE                                        |
| >                                                                    |
| > Training data score: 72.19 MAE                                     |
| >                                                                    |
| > Test data score: 69.09 MAE                                         |
|                                                                      |
| 38                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image45.png){width="5.995832239720035in" |
| height="3.798611111111111in"}                                        |
|                                                                      |
| fig. 37- Comparison of Original and Predicted Time Series(XGBoost    |
| model)                                                               |
|                                                                      |
| > The above figure(fig. 37) shows performance metrics of Xgboost     |
| > model.                                                             |
| >                                                                    |
| > Model :\-- XgBoost                                                 |
| >                                                                    |
| > Training data score: 31.96 RMSE                                    |
| >                                                                    |
| > Test data score: 134.08 RMSE                                       |
| >                                                                    |
| > Training data score: 15.61 MAE                                     |
| >                                                                    |
| > Test data score: 88.06 MAE                                         |
|                                                                      |
| 39                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image46.png){width="5.993054461942258in" |
| height="3.816666666666667in"}                                        |
|                                                                      |
| fig. 38 - Comparison of Original and Predicted Time Series(BiLSTM    |
| model)                                                               |
|                                                                      |
| The above figure(fig. 38) shows performance metrics of LSTM          |
| Bi-Directional model.                                                |
|                                                                      |
| > Model :\-- LSTM Bidirectional                                      |
| >                                                                    |
| > Training data score: 100.96 RMSE                                   |
| >                                                                    |
| > Test data score: 95.44 RMSE                                        |
| >                                                                    |
| > Training data score: 72.67 MAE                                     |
| >                                                                    |
| > Test data score: 69.22 MAE                                         |
|                                                                      |
| 40                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image47.png){width="5.993054461942258in" |
| height="3.8625in"}                                                   |
|                                                                      |
| fig. 39 - Comparison of Original and Predicted Time Series(Stacked   |
| LSTM model)                                                          |
|                                                                      |
| > The above figure(fig. 39) shows performance metrics of LSTM        |
| > Stacked model.                                                     |
| >                                                                    |
| > Model :\-- LSTM Stacked                                            |
| >                                                                    |
| > Training data score: 100.47 RMSE                                   |
| >                                                                    |
| > Test data score: 97.49 RMSE                                        |
| >                                                                    |
| > Training data score: 88.43 MAE                                     |
| >                                                                    |
| > Test data score: 77.66 MAE                                         |
|                                                                      |
| 41                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > From the above plots, It is inferred infer that Elastic net and    |
| > both LSTM modelsbehave similarly and have given stable results.    |
| > The Xgboost model while doesbetter than all model when predicting  |
| > the training dataset, it show instability whenpredicting the test  |
| > dataset.                                                           |
| >                                                                    |
| > The RSME and MAE data of each model have been plotted on a bar     |
| > graph(fig. 40).The conclusion drawn is that the Elastic net        |
| > training model preforms the best abetby a small margin than LSTM   |
| > models. Xgboost shows the problem of Overfitting inits prediction  |
| > set while both LSTM models performs almost same with LSTM          |
| > Bi-Directional model being better of the 2 by a small margin.      |
| >                                                                    |
| > Now that, the best training model for the collected data have been |
| > found. ElasticNet would be used to create Models for other         |
| > sub-divisions.                                                     |
|                                                                      |
| ![](vertopal_a73e3975d707                                            |
| 47a9bedc0205a102929e/media/image48.png){width="5.9944444444444445in" |
| height="4.106944444444444in"}                                        |
|                                                                      |
| fig. 40 - Bar graph comparison of various models using performance   |
| metrics.                                                             |
|                                                                      |
| 42                                                                   |
+----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **8. Sample Code**                                                    |
|                                                                       |
| > **Sample code for Evaluation of dataset and Selection of model**    |
| >                                                                     |
| > import pandas as pd                                                 |
| >                                                                     |
| > import seaborn as sns                                               |
| >                                                                     |
| > from matplotlib import pyplot                                       |
| >                                                                     |
| > import matplotlib.pyplot as plt                                     |
| >                                                                     |
| > import matplotlib as mp                                             |
| >                                                                     |
| > import numpy as np                                                  |
| >                                                                     |
| > import math                                                         |
| >                                                                     |
| > from keras.layers.core import Dense, Activation, Dropout            |
| >                                                                     |
| > from keras.layers import LSTM, Input, Bidirectional                 |
| >                                                                     |
| > from keras.models import Sequential                                 |
| >                                                                     |
| > from xgboost import XGBRegressor                                    |
| >                                                                     |
| > from sklearn import linear_model                                    |
| >                                                                     |
| > from sklearn.metrics import mean_squared_error, mean_absolute_error |
| >                                                                     |
| > from sklearn.preprocessing import MinMaxScaler                      |
| >                                                                     |
| > data = pd.read_csv(                                                 |
| >                                                                     |
| > \'https://raw.githubusercontent.com/\'                              |
| >                                                                     |
| > \'devashishpan/major_project/master/Datasets/\'                     |
| >                                                                     |
| > \'rainfall%20in%20india%201901-2017.csv\'                           |
| >                                                                     |
| > )                                                                   |
|                                                                       |
| 43                                                                    |
+-----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > data.hist(figsize=(24,24));                                        |
| >                                                                    |
| > data.groupby(\"YEAR\").sum()\[\'ANNUAL\'\].plot(figsize=(12,8));   |
| >                                                                    |
| > data\[\[\'YEAR\', \'JAN\', \'FEB\', \'MAR\', \'APR\', \'MAY\',     |
| > \'JUN\', \'JUL\',                                                  |
| >                                                                    |
| > \'AUG\', \'SEP\', \'OCT\', \'NOV\',                                |
| > \'DEC\'\]\].groupby(\"YEAR\").sum().plot(figsize=(13,8));          |
| >                                                                    |
| > plt.figure(figsize=(13,8))                                         |
| >                                                                    |
| > sns.heatmap(data\[\[                                               |
| >                                                                    |
| > \'JAN\',\'FEB\',                                                   |
| >                                                                    |
| > \'MAR\',\'APR\',\'MAY\',                                           |
| >                                                                    |
| > \'JUN\',\'JUL\',\'AUG\',\'SEP\',                                   |
| >                                                                    |
| > \'OCT\',\'NOV\',\'DEC\',\'ANNUAL\'                                 |
| >                                                                    |
| > \]\].corr(),annot=True)                                            |
| >                                                                    |
| > plt.show()                                                         |
| >                                                                    |
| > plt.figure(figsize=(15,8))                                         |
| >                                                                    |
| > sns.heatmap(data.isna(), cmap = \'viridis\');                      |
| >                                                                    |
| > groups = data.groupby(\'SUBDIVISION\')\[\[\'YEAR\',                |
|                                                                      |
| \'JAN\',\'FEB\',\'MAR\',                                             |
|                                                                      |
| \'APR\',\'MAY\',\'JUN\',                                             |
|                                                                      |
| \'JUL\',\'AUG\',\'SEP\',                                             |
|                                                                      |
| \'OCT\',\'NOV\',\'DEC\'                                              |
|                                                                      |
| \]\]                                                                 |
|                                                                      |
| > datajh=groups.get_group((\'Jharkhand\'))                           |
|                                                                      |
| 44                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > datajh=datajh.melt(\[\'YEAR\'\]).reset_index()                     |
| >                                                                    |
| > df=                                                                |
| > datajh\[\[\'index\',\'YEAR                                         |
| \',\'variable\',\'value\'\]\].sort_values(by=\[\'YEAR\',\'index\'\]) |
| >                                                                    |
| > df.columns=\[\'INDEX\',\'YEAR\',\'Month\',\'avg_rainfall\'\]       |
| >                                                                    |
| > d={\'JAN\':1,\'FEB\':2,\'MAR\'                                     |
| > :3,\'APR\':4,\'MAY\':5,\'JUN\':6,\'JUL\':7,\'AUG\':8,\'SEP\':9,    |
| >                                                                    |
| > \'OCT\':10,\'NOV\':11,\'DEC\':12}                                  |
| >                                                                    |
| > df\[\'Month\'\]=df\[\'Month\'\].map(d)                             |
| >                                                                    |
| > cols=\[\]                                                          |
| >                                                                    |
| > dataset=df\[\[\'avg_rainfall\'\]\]                                 |
| >                                                                    |
| > plt.figure(figsize=(15,8))                                         |
| >                                                                    |
| > plt.plot(dataset.values)                                           |
| >                                                                    |
| > plt.show()                                                         |
| >                                                                    |
| > data_raw = dataset.values.astype(\"float32\")                      |
| >                                                                    |
| > m = dataset.mean()\[0\]                                            |
| >                                                                    |
| > for i in range(len(data_raw)):                                     |
| >                                                                    |
| > if(np.isnan(data_raw\[i\])):                                       |
| >                                                                    |
| > data_raw\[i\] = m                                                  |
| >                                                                    |
| > scaler = MinMaxScaler(feature_range = (0, 1))                      |
| >                                                                    |
| > data_scaled = scaler.fit_transform(data_raw)                       |
| >                                                                    |
| > TRAIN_SIZE = 0.80                                                  |
| >                                                                    |
| > train_size = int(len(data_scaled) \* TRAIN_SIZE)                   |
|                                                                      |
| 45                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > test_size = len(data_scaled) - train_size                          |
| >                                                                    |
| > train_dl, test_dl = data_scaled\[0:train_size, :\],\               |
| > data_scaled\[train_size:len(dataset), :\]                          |
|                                                                      |
| print(\"Number of entries (training set, test set): \" +             |
| str((len(train_dl), len(test_dl))))                                  |
|                                                                      |
| > train_size = int(len(data_raw) \* TRAIN_SIZE)                      |
| >                                                                    |
| > test_size = len(data_raw) - train_size                             |
| >                                                                    |
| > train, test = data_raw\[0:train_size, :\],                         |
| > data_raw\[train_size:len(dataset), :\]                             |
| >                                                                    |
| > print(\"Number of entries (training set, test set): \" +           |
| > str((len(train), len(test))))                                      |
| >                                                                    |
| > def create_dataset(dataset, window_size = 1):                      |
| >                                                                    |
| > data_X, data_Y = \[\], \[\]                                        |
| >                                                                    |
| > for i in range(len(dataset) - window_size - 1):                    |
| >                                                                    |
| > a = dataset\[i:(i + window_size), 0\]                              |
| >                                                                    |
| > data_X.append(a)                                                   |
| >                                                                    |
| > data_Y.append(dataset\[i + window_size, 0\])                       |
| >                                                                    |
| > return(np.array(data_X), np.array(data_Y))                         |
| >                                                                    |
| > \# Create test and training sets for one-step-ahead regression.    |
| >                                                                    |
| > window_size = 1                                                    |
| >                                                                    |
| > train_X, train_Y = create_dataset(train, window_size)              |
| >                                                                    |
| > test_X, test_Y = create_dataset(test, window_size)                 |
| >                                                                    |
| > train_X\_, train_Y\_ = create_dataset(train_dl, window_size)       |
| >                                                                    |
| > test_X\_, test_Y\_ = create_dataset(test_dl, window_size)          |
| >                                                                    |
| > print(\"Original training data shape for DL:\")                    |
| >                                                                    |
| > print(train_X\_.shape)                                             |
|                                                                      |
| 46                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > \# Reshape the input data into appropriate form for Keras.         |
| >                                                                    |
| > train_X\_dl = np.reshape(train_X\_, (train_X\_.shape\[0\], 1,      |
| > train_X\_.shape\[1\]))                                             |
| >                                                                    |
| > test_X\_dl = np.reshape(test_X\_, (test_X\_.shape\[0\], 1,         |
| > test_X\_.shape\[1\]))                                              |
| >                                                                    |
| > print(\"New training data shape:\")                                |
| >                                                                    |
| > print(train_X\_dl.shape)                                           |
| >                                                                    |
| > def fit_model_lstm_stacked(train_X, train_Y, window_size = 1):     |
| >                                                                    |
| > model = Sequential()                                               |
| >                                                                    |
| > model.add(Input(shape = (1, window_size)))                         |
| >                                                                    |
| > model.add(                                                         |
| >                                                                    |
| > LSTM(                                                              |
| >                                                                    |
| > 200,                                                               |
| >                                                                    |
| > activation = \'tanh\',                                             |
| >                                                                    |
| > recurrent_activation = \'hard_sigmoid\',                           |
| >                                                                    |
| > return_sequences=True,                                             |
| >                                                                    |
| > )                                                                  |
| >                                                                    |
| > )                                                                  |
| >                                                                    |
| > model.add(Dropout(0.2))                                            |
| >                                                                    |
| > model.add(                                                         |
| >                                                                    |
| > LSTM(                                                              |
| >                                                                    |
| > 200,                                                               |
| >                                                                    |
| > activation = \'tanh\',                                             |
| >                                                                    |
| > recurrent_activation = \'hard_sigmoid\',                           |
| >                                                                    |
| > )                                                                  |
| >                                                                    |
| > )                                                                  |
| >                                                                    |
| > model.add(Dropout(0.4))                                            |
|                                                                      |
| 47                                                                   |
+----------------------------------------------------------------------+

+------------------------------------------------------------------------+
| > model.add(Dense(50,activation = \'tanh\',))                          |
| >                                                                      |
| > model.add(Dropout(0.4))                                              |
| >                                                                      |
| > model.add(Dense(40,activation = \'tanh\',))                          |
| >                                                                      |
| > model.add(Dropout(0.4))                                              |
| >                                                                      |
| > model.add(Dense(1, activation = \'linear\'))                         |
| >                                                                      |
| > model.compile(                                                       |
| >                                                                      |
| > loss = \"mean_squared_error\",                                       |
| >                                                                      |
| > optimizer = \"adam\",                                                |
| >                                                                      |
| > metrics=\[\'accuracy\'\],                                            |
| >                                                                      |
| > )                                                                    |
| >                                                                      |
| > model.fit(train_X,                                                   |
| >                                                                      |
| > train_Y,                                                             |
| >                                                                      |
| > epochs = 10,                                                         |
| >                                                                      |
| > batch_size = 64,                                                     |
| >                                                                      |
| > )                                                                    |
| >                                                                      |
| > return(model)                                                        |
| >                                                                      |
| > def fit_model_lstm_bidirectional(train_X, train_Y, window_size = 1): |
| >                                                                      |
| > model = Sequential()                                                 |
| >                                                                      |
| > model.add(Input(shape = (1, window_size)))                           |
| >                                                                      |
| > model.add(Bidirectional(                                             |
| >                                                                      |
| > LSTM(                                                                |
| >                                                                      |
| > 200,                                                                 |
| >                                                                      |
| > activation = \'tanh\',                                               |
| >                                                                      |
| > recurrent_activation = \'hard_sigmoid\',                             |
| >                                                                      |
| > return_sequences=True,                                               |
| >                                                                      |
| > )                                                                    |
|                                                                        |
| 48                                                                     |
+------------------------------------------------------------------------+

+------------------------------------------------+
| > ))                                           |
| >                                              |
| > model.add(Dropout(0.2))                      |
| >                                              |
| > model.add(Bidirectional(                     |
| >                                              |
| > LSTM(                                        |
| >                                              |
| > 200,                                         |
| >                                              |
| > activation = \'tanh\',                       |
| >                                              |
| > recurrent_activation = \'hard_sigmoid\',     |
| >                                              |
| > )                                            |
| >                                              |
| > ))                                           |
| >                                              |
| > model.add(Dropout(0.4))                      |
| >                                              |
| > model.add(Dense(50,activation = \'tanh\',))  |
| >                                              |
| > model.add(Dropout(0.4))                      |
| >                                              |
| > model.add(Dense(40,activation = \'tanh\',))  |
| >                                              |
| > model.add(Dropout(0.4))                      |
| >                                              |
| > model.add(Dense(1, activation = \'linear\')) |
| >                                              |
| > model.compile(                               |
| >                                              |
| > loss = \"mean_squared_error\",               |
| >                                              |
| > optimizer = \"adam\",                        |
| >                                              |
| > metrics=\[\'accuracy\'\],                    |
| >                                              |
| > )                                            |
| >                                              |
| > model.fit(train_X,                           |
| >                                              |
| > train_Y,                                     |
| >                                              |
| > epochs = 10,                                 |
| >                                              |
| > batch_size = 64,                             |
| >                                              |
| > )                                            |
| >                                              |
| > return(model)                                |
|                                                |
| 49                                             |
+------------------------------------------------+

+----------------------------------------------------------------------+
| > def fit_model_elasticnet(train_X, train_Y):                        |
| >                                                                    |
| > model = linear_model.ElasticNet(alpha=0.5)                         |
| >                                                                    |
| > model.fit(train_X, train_Y)                                        |
| >                                                                    |
| > return(model)                                                      |
| >                                                                    |
| > def fit_model_xgboost(train_X, train_Y):                           |
| >                                                                    |
| > model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1,      |
| >                                                                    |
| > subsample=0.7)                                                     |
| >                                                                    |
| > model.fit(train_X, train_Y)                                        |
| >                                                                    |
| > return(model)                                                      |
| >                                                                    |
| > model_lstm_stacked = fit_model_lstm_stacked(train_X\_dl,           |
| > train_Y\_, window_size)                                            |
| >                                                                    |
| > model_lstm_bidirectional =                                         |
| > fit_model_lstm_bidirectional(train_X\_dl, train_Y\_, window_size)  |
| >                                                                    |
| > model_elasticnet = fit_model_elasticnet(train_X,train_Y)           |
| >                                                                    |
| > model_xgboost = fit_model_xgboost(train_X,train_Y)                 |
| >                                                                    |
| > models = {                                                         |
| >                                                                    |
| > \"Elastic Net\" : model_elasticnet,                                |
| >                                                                    |
| > \"XgBoost\" : model_xgboost                                        |
| >                                                                    |
| > }                                                                  |
| >                                                                    |
| > models_dl = {                                                      |
| >                                                                    |
| > \"LSTM Bidirectional\" : model_lstm_bidirectional,                 |
| >                                                                    |
| > \"LSTM Stacked\" : model_lstm_stacked                              |
| >                                                                    |
| > }                                                                  |
| >                                                                    |
| > def plot_model_dl(train_predict,test_predict):                     |
| >                                                                    |
| > train_predict_plot = np.empty_like(data_scaled)                    |
|                                                                      |
| 50                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > train_predict_plot\[:, :\] = np.nan                                |
| >                                                                    |
| > train_predict_plot\[window_size:len(train_predict) + window_size,  |
| > :\] =                                                              |
| >                                                                    |
| > train_predict                                                      |
| >                                                                    |
| > test_predict_plot = np.empty_like(data_scaled)                     |
| >                                                                    |
| > test_predict_plot\[:, :\] = np.nan                                 |
| >                                                                    |
| > test_predict_plot\[                                                |
| >                                                                    |
| > len(train_predict) + (window_size \* 2) + 1:len(dataset) - 1, :\]  |
| > = test_predict                                                     |
| >                                                                    |
| > plt.figure(figsize = (15, 8))                                      |
| >                                                                    |
| > plt.plot(scaler.inverse_transform(data_scaled), label = \"True     |
| > value\",color=\'blue\')                                            |
| >                                                                    |
| > plt.plot(train_predict_plot , label = \"Training set               |
| > prediction\",color=\'yellow\')                                     |
| >                                                                    |
| > plt.plot(test_predict_plot, label = \"Test set                     |
| > prediction\",color=\'red\')                                        |
| >                                                                    |
| > plt.xlabel(\"Months\")                                             |
| >                                                                    |
| > plt.legend()                                                       |
| >                                                                    |
| > plt.show()                                                         |
| >                                                                    |
| > def plot_model(train_predict,test_predict):                        |
| >                                                                    |
| > train_predict_plot =                                               |
| > np.empty_like(data_raw.reshape(len(data_raw),))                    |
| >                                                                    |
| > train_predict_plot\[:\] = np.nan                                   |
| >                                                                    |
| > train_predict_plot\[window_size:len(train_predict) + window_size\] |
| > = train_predict                                                    |
| >                                                                    |
| > test_predict_plot =                                                |
| > np.empty_like(data_raw.reshape(len(data_raw),))                    |
| >                                                                    |
| > test_predict_plot\[:\] = np.nan                                    |
|                                                                      |
| 51                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > test_predict_plot\[                                                |
|                                                                      |
| len(train_predict) + (window_size \* 2) + 1:len(dataset) - 1\] =     |
| test_predict                                                         |
|                                                                      |
| > plt.figure(figsize = (15, 8))                                      |
| >                                                                    |
| > plt.plot(data_raw, label = \"True value\",color=\'blue\')          |
| >                                                                    |
| > plt.plot(train_predict_plot , label = \"Training set               |
| > prediction\",color=\'yellow\')                                     |
| >                                                                    |
| > plt.plot(test_predict_plot, label = \"Test set                     |
| > prediction\",color=\'red\')                                        |
| >                                                                    |
| > plt.xlabel(\"Months\")                                             |
| >                                                                    |
| > plt.legend()                                                       |
| >                                                                    |
| > plt.show()                                                         |
| >                                                                    |
| > def predict_and_score(model, X, Y):                                |
| >                                                                    |
| > pred = model.predict(X)                                            |
| >                                                                    |
| > orig_data = Y                                                      |
| >                                                                    |
| > score = math.sqrt(mean_squared_error(orig_data, pred\[:\]))        |
| >                                                                    |
| > mean = mean_absolute_error(orig_data, pred\[:\])                   |
| >                                                                    |
| > return(score, pred, mean)                                          |
| >                                                                    |
| > def score(models):                                                 |
| >                                                                    |
| > for model in models.keys():                                        |
|                                                                      |
| rmse_train, train_predict, mae_train =                               |
| predict_and_score(models\[model\],                                   |
|                                                                      |
| > train_X, train_Y)                                                  |
| >                                                                    |
| > rmse_test, test_predict, mae_test =                                |
| > predict_and_score(models\[model\], test_X,                         |
| >                                                                    |
| > test_Y)                                                            |
| >                                                                    |
| > print(f\"Model :\-- {model}\")                                     |
| >                                                                    |
| > print(\"Training data score: %.2f RMSE\" % rmse_train)             |
| >                                                                    |
| > print(\"Test data score: %.2f RMSE\" % rmse_test)                  |
| >                                                                    |
| > print(\"Training data score: %.2f MAE \" % mae_train)              |
|                                                                      |
| 52                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > print(\"Test data score: %.2f MAE \" % mae_test)                   |
| >                                                                    |
| > Scores\[model\] = \[rmse_test,rmse_train,mae_test,mae_train\]      |
| >                                                                    |
| > plot_model(train_predict,test_predict)                             |
| >                                                                    |
| > print()                                                            |
| >                                                                    |
| > def predict_and_score_dl(model, X, Y):                             |
| >                                                                    |
| > pred = scaler.inverse_transform(model.predict(X))                  |
| >                                                                    |
| > orig_data = scaler.inverse_transform(\[Y\])                        |
| >                                                                    |
| > score = math.sqrt(mean_squared_error(orig_data\[0\], pred\[:,      |
| > 0\]))                                                              |
| >                                                                    |
| > mean = mean_absolute_error(orig_data\[0\], pred\[:, 0\])           |
| >                                                                    |
| > return(score, pred, mean)                                          |
| >                                                                    |
| > def score_dl(models):                                              |
| >                                                                    |
| > for model in models.keys():                                        |
| >                                                                    |
| > rmse_train, train_predict, mae_train =                             |
| > predict_and_score_dl(models\[model\],                              |
| >                                                                    |
| > train_X\_dl, train_Y\_)                                            |
|                                                                      |
| rmse_test, test_predict, mae_test =                                  |
| predict_and_score_dl(models\[model\],                                |
|                                                                      |
| > test_X\_dl, test_Y\_)                                              |
| >                                                                    |
| > print(f\"Model :\-- {model}\")                                     |
| >                                                                    |
| > print(\"Training data score: %.2f RMSE\" % rmse_train)             |
| >                                                                    |
| > print(\"Test data score: %.2f RMSE\" % rmse_test)                  |
| >                                                                    |
| > print(\"Training data score: %.2f MAE \" % mae_train)              |
| >                                                                    |
| > print(\"Test data score: %.2f MAE \" % mae_test)                   |
| >                                                                    |
| > Scores\[model\] = \[rmse_test,rmse_train,mae_test,mae_train\]      |
| >                                                                    |
| > plot_model_dl(train_predict,test_predict)                          |
| >                                                                    |
| > print()                                                            |
|                                                                      |
| 53                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > score(models)                                                      |
| >                                                                    |
| > score_dl(models_dl)                                                |
| >                                                                    |
| > d = pd.DataFrame(Scores)                                           |
| >                                                                    |
| > y = d.transpose().reset_index()                                    |
| >                                                                    |
| > y.columns =                                                        |
| > \                                                                  |
| [\'Model\',\'RMSE_Test\',\'RSME_Train\',\'MAE_Test\',\'MAE_Train\'\] |
| >                                                                    |
| > y.plot(x=\"Mo                                                      |
| del\",y=\[\'RMSE_Test\',\'RSME_Train\',\'MAE_Test\',\'MAE_Train\'\], |
| > kind=\"bar\",figsize=(12,8));                                      |
|                                                                      |
| 54                                                                   |
+----------------------------------------------------------------------+

+--------------------------------------------------------------------------+
| > **Sample Code for prediction model creation using the selected model** |
| >                                                                        |
| > import pandas as pd                                                    |
| >                                                                        |
| > from sklearn import linear_model                                       |
| >                                                                        |
| > import joblib as jb                                                    |
| >                                                                        |
| > import numpy as np                                                     |
| >                                                                        |
| > data = pd.read_csv(                                                    |
| >                                                                        |
| > \'https://raw.githubusercontent.com/\'                                 |
| >                                                                        |
| > \'devashishpan/major_project/master/Datasets/\'                        |
| >                                                                        |
| > \'rainfall%20in%20india%201901-2017.csv\'                              |
| >                                                                        |
| > )                                                                      |
| >                                                                        |
| > subdivisions = data.SUBDIVISION.unique()                               |
| >                                                                        |
| > groups = data.groupby(\'SUBDIVISION\')\[\[\'YEAR\',                    |
|                                                                          |
| \'JAN\',\'FEB\',\'MAR\',                                                 |
|                                                                          |
| \'APR\',\'MAY\',\'JUN\',                                                 |
|                                                                          |
| \'JUL\',\'AUG\',\'SEP\',                                                 |
|                                                                          |
| \'OCT\',\'NOV\',\'DEC\'                                                  |
|                                                                          |
| \]\]                                                                     |
|                                                                          |
| > def create_dataset(dataset, window_size = 1):                          |
| >                                                                        |
| > data_X, data_Y = \[\], \[\]                                            |
| >                                                                        |
| > for i in range(len(dataset) - window_size - 1):                        |
| >                                                                        |
| > a = dataset\[i:(i + window_size), 0\]                                  |
| >                                                                        |
| > data_X.append(a)                                                       |
| >                                                                        |
| > data_Y.append(dataset\[i + window_size, 0\])                           |
|                                                                          |
| 55                                                                       |
+--------------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > return(np.array(data_X), np.array(data_Y))                         |
| >                                                                    |
| > def fit_model_elasticnet(train_X, train_Y):                        |
| >                                                                    |
| > model = linear_model.ElasticNet(alpha=0.5)                         |
| >                                                                    |
| > model.fit(train_X, train_Y)                                        |
| >                                                                    |
| > return(model)                                                      |
| >                                                                    |
| > d={\'JAN\':1,\'FEB\':2,\'MAR\'                                     |
| > :3,\'APR\':4,\'MAY\':5,\'JUN\':6,\'JUL\':7,\'AUG\':8,\'SEP\':9,    |
| >                                                                    |
| > \'OCT\':10,\'NOV\':11,\'DEC\':12}                                  |
| >                                                                    |
| > TRAIN_SIZE = 0.80                                                  |
| >                                                                    |
| > window_size = 1                                                    |
| >                                                                    |
| > def create_models(groups):                                         |
| >                                                                    |
| > for i in subdivisions:                                             |
| >                                                                    |
| > temp = groups.get_group((i))                                       |
| >                                                                    |
| > temp = temp.melt(\[\'YEAR\'\]).reset_index()                       |
| >                                                                    |
| > temp =                                                             |
| >                                                                    |
| > temp\[\[\'index\',\'YEAR                                           |
| \',\'variable\',\'value\'\]\].sort_values(by=\[\'YEAR\',\'index\'\]) |
| >                                                                    |
| > temp.columns=\[\'INDEX\',\'YEAR\',\'Month\',\'avg_rainfall\'\]     |
| >                                                                    |
| > temp\[\'Month\'\] = temp\[\'Month\'\].map(d)                       |
| >                                                                    |
| > dataset=temp\[\[\'avg_rainfall\'\]\]                               |
| >                                                                    |
| > m = dataset.mean()\[0\]                                            |
| >                                                                    |
| > data_raw = dataset.values.astype(\"float32\")                      |
| >                                                                    |
| > for i in range(len(data_raw)):                                     |
| >                                                                    |
| > if(np.isnan(data_raw\[i\])):                                       |
|                                                                      |
| 56                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > data_raw\[i\] = m                                                  |
| >                                                                    |
| > train_size = int(len(data_raw) \* TRAIN_SIZE)                      |
| >                                                                    |
| > test_size = len(data_raw) - train_size                             |
|                                                                      |
| train, test = data_raw\[0:train_size, :\],                           |
| data_raw\[train_size:len(dataset), :\]                               |
|                                                                      |
| > train_X, train_Y = create_dataset(train, window_size)              |
| >                                                                    |
| > test_X, test_Y = create_dataset(test, window_size)                 |
| >                                                                    |
| > model_elasticnet = fit_model_elasticnet(train_X,train_Y)           |
| >                                                                    |
| > jb.dump(model_elasticnet,f\"models/{i}.model\")                    |
| >                                                                    |
| > create_models(groups)                                              |
| >                                                                    |
| > The above code will create and save all Prediction models for all  |
| > sub-divisions inmodels directory.                                  |
|                                                                      |
| 57                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| **9. Sample Output**                                                 |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image49.png){width="7.001388888888889in" |
| height="3.4430555555555555in"}                                       |
|                                                                      |
| fig. 41 - Models Created by Sample Code.                             |
|                                                                      |
| > The above output models(fig. 41) can then be uploaded on Github or |
| > any otherrepository and any body can download and use them as they |
| > seem fit.                                                          |
| >                                                                    |
| > The models can easily be loaded(fig. 42 & 43) and then used to     |
| > predict rainfall:-                                                 |
|                                                                      |
| ![](vertopal_a73e3975d70                                             |
| 747a9bedc0205a102929e/media/image50.png){width="6.283333333333333in" |
| height="2.841666666666667in"}                                        |
|                                                                      |
| fig. 42 - Importing and using Jharkhand Model for Prediction.        |
|                                                                      |
| 58                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| ![](vertopal_a73e3975d707                                            |
| 47a9bedc0205a102929e/media/image51.png){width="5.9944444444444445in" |
| height="4.981944444444444in"}                                        |
|                                                                      |
| fig. 43 - Prediction and real value graphs for comparison.           |
|                                                                      |
| 59                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| **10. Conclusion & Future Scope**                                    |
|                                                                      |
| > Rainfall being one of the sole responsibilities for maximum        |
| > economy of India, itshould be considered the primary concern for   |
| > most of us. Currently, rainfallprediction has become one of the    |
| > key factors for most of the water conservationsystems in and       |
| > across country. One of the biggest challenges is the               |
| > complexitypresent in rainfall data. Most of the rainfall           |
| > prediction system, nowadays areunable to find the hidden features  |
| > or any non-linear patterns present in the datadue to use of old    |
| > statistical methods.                                               |
| >                                                                    |
| > Machine Learning approach can been used to tackle the issue. The   |
| > Dataset of4116 rows has used that is converted to month-wise       |
| > time-series data of rainfall invarious regions in and around the   |
| > country. A small part of the data was consideredfor test run i.e   |
| > of the various regions(sub-divisions)only one is used.             |
| > Theconsidered subset is divided into two parts i.e. train data and |
| > test data(80:20 split).Train data is used for training the         |
| > algorithm and test data is for doing theverification and           |
| > evaluation. 4 models - 2 Machine Learning models (Elastic Net      |
| > &XGBoost) and 2 Deep Learning Neural Network models(Stacked LSTM & |
| > BiLSTM)have been configured to create the predictor models. All    |
| > models are thencompared based on their performance metrics(RMSE    |
| > and MAE). The one withbetter performance metrics is considered and |
| > implemented on the rest of the datato create models for each       |
| > region. We concluded that Elastic Net regressionmethods was better |
| > than other approaches and models for each sub-division iscreated.  |
| >                                                                    |
| > Currently the model can predict the rainfall for next month using  |
| > the rainfall dataof current month. This can further be improved in |
| > future so that prediction can bemade for a each day or may be for  |
| > an year in advance. The constraint to suchimprovement is basically |
| > the dataset. The current dataset is only provides us withmonthly   |
| > data only.                                                         |
| >                                                                    |
| > A better data-set with daily entries if produced can enhance the   |
| > model creation tobe able to predict the amount to rainfall for     |
| > each consecutive day. Similarly, the                               |
|                                                                      |
| 60                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| > model could be enhanced to be able to predict rainfall amount a    |
| > year in advance,by using Multistep-ahead forecasting methods, in a |
| > way that inputting the monthlyrainfall data of each month of       |
| > current year will predict the rainfall for the monthsof the next   |
| > year.                                                              |
|                                                                      |
| 61                                                                   |
+----------------------------------------------------------------------+

+----------------------------------------------------------------------+
| **11. Bibliography**                                                 |
|                                                                      |
| > 1\. Praveen, B., Talukdar, S., Shahfahad et al. Analyzing trend    |
| > and forecasting ofrainfall changes in India using non-parametrical |
| > and machine learning approaches.                                   |
| >                                                                    |
| > Sci Rep 10, 10342 (2020).                                          |
| > https://doi.org/10.1038/s41598-020-67228-7                         |
| >                                                                    |
| > 2\. Demeke Endalie, Getamesay Haile, Wondmagegn Taye; Deep         |
| > learning modelfor daily rainfall prediction: case study of Jimma,  |
| > Ethiopia. Water Supply 1 March2022; 22 (3): 3448--3461. doi:       |
| > https://doi.org/10.2166/ws.2021.391                                |
| >                                                                    |
| > 3\. B.Meena Preethi, R. Gowtham, S.Aishvarya, S.Karthick,          |
| > D.G.Sabareesh,.ainfallPrediction using Machine Learning and Deep   |
| > Learning Algorithms. DOI link:                                     |
| >                                                                    |
| > 4\. Machine Learning Geeks Of Geeks .                              |
| > Link:-https://www.geeksforgeeks.org/machine-learning/              |
|                                                                      |
| 62                                                                   |
+----------------------------------------------------------------------+
