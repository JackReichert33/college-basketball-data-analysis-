# college-basketball-data-analysis-
The main purpose for this project will be to examine which team statistics are most strongly associated with NCAA tournament success. The dataset being studied is made up of data from the 2013 through 2026 Division I college basketball seasons.

Research Question: Which team statistics are most strongly associated with NCAA Tournament Success? 

Dataset: https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset?resource=download 
kagglehub.dataset_download('andrewsundberg/college-basketball-dataset') 

Number of Observations:  (4249, 24) 

Introduction 

The NCAA Men’s Basketball Tournament, the widely followed and beloved sporting event known as March Madness, represents one of the few occasions when the United States comes together, focusing on the enticing combination of college basketball and socially acceptable gambling; it’s an opportunity for both devoted fans and the uninitiated to come together in the joy and heartbreak that is selecting their March Madness brackets, using a wide variety of selection criteria.  While seeding, team reputation, and personal affiliations often influence selections, statistical performance analysis may provide a more objective basis for bracket choices. 

 

Project Description 

This project will analyze team performance data from Division I college basketball seasons 2013 through 2026.  The goal is to identify which statistical measures are most associated with tournament advancement and determine if the predictors are consistent over time.  The project's focus is evaluating offensive, defensive, efficiency, and ranking metrics to determine their relationship to postseason success.  Machine learning models and statistical analysis techniques will compare the predictive value of various performance indicators. 





# METHODS SECTION


(These Steps Below represent a logistic regression model where all variables 	are tied together)  

# 1st Step  

The first step in the logistic regression model process was to prepare the dataset in 	a way that identified a target variable. This was first done by keeping teams in the dataset with recorded NCAA Tournament results using the line of code: dropna (subset=["POSTSEASON”]) which gave us 816 tournament teams to work with. Next, a binary target variable called ADVANCED_R64 was created, and a value of 1 was assigned to teams who advanced (384) and 0 was assigned to team who did not advance (432). 

# 2nd Step  

Next,  7 statistical variables were selected, and the data were trained and tested. The seven variables include BARTHAG (probability of beating an average Division 1 team),  ADJOE (Adjusted offensive efficiency), ADJDE (Adjusted defensive efficiency), EFGO (Offensive effective field goal percentage), EFGD (Defensive filed goal percentage), WAB (Wins above Bubble), and SEED (NCAA Tournament seed). Once the data was checked for missing values using the expression isnull().sum(), the data was slit into 80 percent training data and 20 percent testing data. 

 

# 3rd Step  

 

In the 3rd Step, the logistic regression model was built and trained. To do this, 3 main tools were used that included StandardScaler() which standardized the seven basketball statistics so they were on comparable scales, LogisticRegression(maxiter=1000), which created the logistic regression model, and Pipleline(), which combined the standardization and logistic regression into one process. After this mode. fit(Xtrain, ytrain), trained the model with 652 observations. Once the training was completed, ypred = model.Predict(X_test) generated 164 predictions for the 164 teams in the testing dataset. 

 

# 4th Step  

The next step was to properly evaluate the model’s predictive performance. 	The model had  a 73.17% accuracy rating, with 63 teams correctly predicted to not 	advance with 24 incorrectly predicted to advanand 57 teams correctly predicted to 	advance, and 20 correctly predicted to not advance. 

Additionally, evaluation metrics included a precision of 69%, recall of 74%, and F-1 score of 71%.  Precision tell us how often the model was correct when predicting a team would advance, recall represents how many of the teams that actually advanced and the model successfully identified, and F-1 score is a combined measurement of precision and recall. 

 

# 5th Step  

The next step consisted of examining coefficients and ranking the predictors. These coefficients were extracted from the trained logistic regression model and were organized into a table. The positive coefficients indicated higher modeled odds of advancing while the negative coefficients indicated lower modeled odds of advancing. The absolute coefficient value shows the overall magnitude of the estimated relationship, and the largest overall coefficients were ADJOE and ADJDE.   

 

# 6th Step  

The last step was arguably the most important part when answering the research question of which team statistics are most associated with NCAA tournament success. This step including calculating and interpreting the odds ratios. To convert the logistic regression coefficients into odds ratios the expression, np.exp(coefficients_df[“Coefficient”]) was used. Overall, ADJOE had the largest positive odds ratio, and therefore was associated with substantially higher odds of advancing. A higher ADJDE was linked with lower odds of advancing, with one standard deviation increase being approximately 63.1% for lower odds. 

 

In addition to analyzing the variable features all in one, analyzing the variables separately was important to be able to properly answer our research question. The only distinct change made to the model was not include all seven variables together simultaneously, while maintaining ADVANCED_R64 as the dependent variable.  

 

Example Input from seven variable logistic regression: 

#Building the seven predictor variables that we chose to select, leaving out postseason and 	year. 

features = (["BARTHAG", "ADJOE", "ADJDE", "EFG_O", "EFG_D", "WAB", "SEED"]) 

X= tournament_df[features] 

y= tournament_df["ADVANCED_R64"] 

 

Example Input from single variable logistic regression:  

 

# Selecting just one predictor 

X_train_single = X_train[[feature]] 

X_test_single = X_test[[feature]] 

 

 


# RESULTS SECTION
 The most interesting finding came from the BARTHAG feature representing a team’s overall strength (ability to beat an average Division 1 team on a netural court). BARTHAG has an odds ratio of 0.695 , which represented a (1-0.695) x 100 = 30.5% lower odds of advancing with a one standard deviation increase. This is considered puzzling due to it being expected that stronger teams with a high BARTHAG would be more likely to advance, but the table shows the opposite.  

To further investigate the unexpected relationship with BARTHAG and tournament advancement, a correlation matrix was created to better examine relationships between selected predictors. The results showed that BARTHAG had a strong positive correlation with WAB (0.90) and ADJOE (0.77) (as one variable increases, the other tends to Increase), along with a strong negative correlation with SEED (-0.83) and ADJDE (-0.77) (as one variable increases, the other tends to decrease). These relationships explain that several predictors being used overlap aspects of team strength. Therefore, this might explain why BARTHAG produced a negative coefficient within the logistic regression model. Overall, this means that the negative odds ratio should be interpreted cautiously.  

    Feature    Coefficient      Absolute Coefficient Value     Odds Ratio 

1    ADJOE      1.096938                 1.096938                 2.994981 

2    ADJDE      -0.997976                0.997976                 0.368625 

0   BARTHAG    -0.364015                 0.364015                 0.694880 

6     SEED     -0.356648                 0.356648                 0.700019 

5      WAB     0.220043                  0.220043                1.246130 

4    EFG_D     0.152706                  0.152706                 1.164982 

3    EFG_O    -0.050602                  0.050602                 0.950657 

 

After realizing that the negative BARTHAG coefficient might be influenced by the other predictor variables, a second logistic regression model was created and 	made the exact same, just leaving out the BARTHAG variable. The largest changes occurred in ADJOE and ADJDE. ADJOE’s odds ratio decreased from 2.995 to 2.478, 	and ADJDE’s odd ratio increased from 0.369 to 0.445. Therefore, this proves that 	BARTHAG had a strong positive relationship with ADJOE and a strong negative relationship with the ADJDE variable.  

 
 
 Feature  Coefficient    Absolute Coefficient Value   Odds Ratio 

0   ADJOE     0.907573    0.907573                     2.478300 

1   ADJDE    -0.810712    0.810712                     0.444542 

5    SEED    -0.379596    0.379596                     0.684138 

4     WAB     0.177804    0.177804                     1.194591 

3   EFG_D     0.143024    0.143024                     1.153757 

2   EFG_O    -0.040873    0.040873                     0.959951 

 

 

 

 

 

 

 

 

 

 

# LOGISTIC REGRESSION MODEL WITH THE VARIABLES NOT TIED TOGETHER  

 

 Feature  Coefficient  Odds Ratio 

0   ADJOE     1.272996    3.571537 

1   ADJDE    -1.057045    0.347481 

2   EFG_O     0.322332    1.380343 

3   EFG_D    -0.407191    0.665517 

4     WAB     1.687114    5.403861 

5    SEED    -1.452620    0.233956 

 

This model identified WAB as having the strongest association with 	advancing beyond the Round of 64, followed by Seed, ADJOE, and ADJDE. Overall, these results suggest that overall team performance (WAB), tournament seeding, and offensive and defensive efficiency are each important individual variables when determining tournament advancement.  

 

