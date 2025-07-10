---
title: ""
output: pdf_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```



--------------------------------------------------------------------------------------

## Problem 1

Use the wbca data(copy and paste the following command into R console:library(faraway);data("wbca"). The dataset wbca comes from a study of breast cancer in Wisconsin. There are 681 cases of potentially cancerous tumors of which 238 are actually malignant. Determining whether a tumor is really malignant is traditionally determined by an invasive surgical procedure. The purpose of this study was to determine whether a new procedure called fine needle aspiration, which draws only a small sample of tissue, could be effective in determining tumor status. Use 70% of the data for train data set and use remaining data (30% of data) for test data set (use set.seed(1023)). (60 points, 10 points each)

```{r}
library(faraway)
data("wbca")
q1.dat=wbca
#q1.dat<-na.omit(q1.dat[-1])

set.seed(1234)
n1=nrow(q1.dat)
q1_sample <- sample(n1, 0.7*n1)
q1_train_data <- q1.dat[q1_sample, ]
q1_test_data <- q1.dat[-q1_sample, ]
```


a-) Fit a binary regression with Class as the response and the other nine variables as predictors on the train data set. Report the residual deviance and associated degrees of freedom. Can this information be used to determine if this model fits the data? Explain.

```{r}
g<-glm(Class~.,data=q1_train_data,family=binomial)
g1<-step(g,trace=0)
g1$formula
```

\textcolor{blue}{BNucl + Chrom + NNucl + Thick + UShap are significant.}

```{r}
g1<-glm(Class ~ ., data=q1_train_data, family=binomial)
summary(g1)
```


b-) Use AIC as the criterion to determine the best subset of variables. (Use the step function.)

```{r}
library(olsrr)
set.seed(1023)
# Find the model using AIC criteria
ols_step_backward_aic(g1)
```

```{r}
# The model with selected variables according to AIC 
#glm(Class ~ .-Mitos -Adhes , data=q1_train_data, family=binomial)
reg1b<- glm(Class~.-Mitos -Adhes , data=q1_train_data, family=binomial)
#summary(reg1b) we note that under this approach several variables with p-value above 5% were selected, they may be removed
ols_aic(reg1b)

summary(reg1b)
```

\textcolor{blue}{So by best AIC the model contains only 7 variables.}

c-) Suppose that a cancer is classified as benign if p > 0.5 and malignant if p < 0.5. Compute the confusion matrix and accuracy rates if this method is applied to the train data with the reduced model. 

```{r}
library(caret)
predicted_probabilities <- reg1b$fitted.values
true_labels<- ifelse(q1_train_data$Class=="1", 1, 0)
predictions <- ifelse(predicted_probabilities > 0.5, 1, 0)
confusion_matrix <- confusionMatrix(as.factor(predictions),
                                    as.factor(true_labels),
                                    mode="prec_recall",positive = "1")
confusion_matrix
```

\textcolor{blue}{F1 statistic is 0.9885, the accuracy ratio is 0.9853 and recall is 0.9869. The model only misclassified 4 benign cases as malignant; and 3 malignant cases as benign. The overall model performance is satisfactory. Remaining 469 cases were classified correctly (167 malignant, and 302 benign).}

d-) Suppose we change the cutoff to 0.9 so that p < 0.9 is classified as malignant and p > 0.9 as benign. Compute the confusion matrix.

```{r}
library(caret)
predicted_probabilities <- reg1b$fitted.values
true_labels<- ifelse(q1_train_data$Class=="1", 1, 0)
predictions <- ifelse(predicted_probabilities > 0.9, 1, 0)
confusion_matrix <- confusionMatrix(as.factor(predictions),
                                    as.factor(true_labels),
                                    mode="prec_recall",positive = "1")
confusion_matrix
```

\textcolor{blue}{F1 statistic is 0.9834, the accuracy ratio is 0.979 and recall is 0.9706. The model only misclassified 9 benign cases as malignant; and 1 malignant cases as benign. The overall model performance is satisfactory. Remaining 466 cases were classified correctly (169 malignant, and 297 benign).}

e-) Perform Hosmer-Lemeshow Goodness of Fit Test and comment

```{r}
library(ResourceSelection)
# 5 is forming 5 groups
hoslem.test(reg1b$y,fitted(reg1b),g=5)
```


\textcolor{blue}{Ho: Fit is good. Ha: Fit is not good. As p-value=0.9792>0.05, so accept null hypothesis and fit is good.}

f-) Repeat par c and d on the test data set and comment on the model performance 

```{r}
library(caret)
predicted_probabilities <- predict(reg1b,q1_test_data,type="response")
true_labels<- ifelse(q1_test_data$Class=="1", 1, 0)
predictions <- ifelse(predicted_probabilities > 0.5, 1, 0)
confusion_matrix <- confusionMatrix(as.factor(predictions),
                                    as.factor(true_labels),
                                    mode="prec_recall",positive = "1")
confusion_matrix
```

\textcolor{blue}{Repeat c using test data, F1 statistic is 0.9451, the accuracy ratio is 0.9268 and recall is 0.9416. The model only misclassified 8 benign cases as malignant; and 7 malignant cases as benign. The overall model performance is satisfactory. Remaining 190 cases were classified correctly (61 malignant, and 129 benign). The testing data set has less prediction accuracy comparing to training data set.}

```{r}
library(caret)
predicted_probabilities <- predict(reg1b,q1_test_data,type="response")
true_labels<- ifelse(q1_test_data$Class=="1", 1, 0)
predictions <- ifelse(predicted_probabilities > 0.9, 1, 0)
confusion_matrix <- confusionMatrix(as.factor(predictions),
                                    as.factor(true_labels),
                                    mode="prec_recall",positive = "1")
confusion_matrix
```

\textcolor{blue}{Repeat on d using testing data, F1 statistic is 0.9509, the accuracy ratio is 0.9366 and recall is 0.9197. The model only misclassified 11 benign cases as malignant; and 2 malignant cases as benign. The overall model performance is satisfactory. Remaining 192 cases were classified correctly (66 malignant, and 126 benign). The testing data set has lower prediction accuracy comparing to training data set.}

## Problem 2

Use train and test data set in problem 1 to answer the questions below.(40 points, 10 points each)

a-) Fit a decision tree model to predict Class by using the other nine variables as predictors on the train data set. Comment on the decision tree.

```{r}
set.seed(1023)
# Fitting the tree on train data set
par(mfrow=c(1,1))
library(rpart)
```

```{r}
# get training decision tree
q2.tree <- rpart(Class ~ ., data = q1_train_data)
library(rpart.plot)
rpart.plot(q2.tree, digits = 3)
```

```{r}
#performance of regression tree on train data
PredictedTest<-predict(q2.tree,q1_train_data)
ModelTest2<-data.frame(obs = q1_train_data$Class, pred=PredictedTest)
defaultSummary(ModelTest2)
```

\textcolor{blue}{$R^2$ on the train data set is 90% and the model performance is stable on the
train data sets.}




b-) Calculate the confusing matrix on the train data set and compare against the logistic regression model confusion matrix on the same.

```{r}
library(C50)
set.seed(1234)
# delete 1st column i.e., Class
a=q1_train_data[-1]
g2 <- C5.0(q1_train_data[-1], as.factor(q1_train_data$Class))
true_labels<- ifelse(q1_train_data$Class=="1", 1, 0)
predicted_class <- predict(g2, q1_train_data)
predicted_labels<- ifelse(predicted_class=="1", 1, 0)
confusion_matrix <- confusionMatrix(as.factor(predicted_labels), as.factor(true_labels),mode="prec_recall", positive = "1")
confusion_matrix

```

\textcolor{blue}{Per training data, F1 statistic is 0.9951, the accuracy ratio is 0.9937 and recall is 100%. The model only misclassified 0 malignant cases as benign; and 3 benign cases as malignant. The overall model performance is satisfactory. Remaining 473 cases were classified correctly (167 malignant, and 306 benign). Decision tree model performs better to logistic regression on the train data set.}


c-) Calculate the confusion matrix on the test data set and compare this against the logistic regression model output on the test data set. 

```{r}
library(C50)
set.seed(1234)
# delete 1st column i.e., Class
a=q1_test_data[-1]
g2 <- C5.0(q1_test_data[-1], as.factor(q1_test_data$Class))
true_labels<- ifelse(q1_test_data$Class=="1", 1, 0)
predicted_class <- predict(g2, q1_test_data)
predicted_labels<- ifelse(predicted_class=="1", 1, 0)
confusion_matrix <- confusionMatrix(as.factor(predicted_labels), as.factor(true_labels),mode="prec_recall", positive = "1")
confusion_matrix

```

\textcolor{blue}{Per testing data, F1 statistic is 0.9814, the accuracy ratio is 0.9756 and recall is 0.9635. The model only misclassified 5 malignant cases as benign; and 0 benign cases as malignant. The overall model performance is satisfactory. Remaining 200 cases were classified correctly (68 malignant, and 132 benign). Decision tree model performs better to logistic regression on the testing data set.}

d-) Which model would you choose?

\textcolor{blue}{I prefer decision tree model as it has higher prediction accuracy in both training and testing data sets.}
