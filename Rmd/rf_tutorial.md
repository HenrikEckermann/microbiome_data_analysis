Random Forest and Extremely Randomized Trees
============================================

The random forest algorithm (RF) is an advanced tree based method that
can be utilized for both classification and regression. Tree methods are
a great data analysis tools as they are:

-   robust against outliers
-   robust (to some extend) to irrelevant variables
-   invariant to scaling of inputs
-   computationally efficient and scalable
-   appropriate for high dimensional data
-   able to predict non-linear relationships

The simplest tree based method would be a single decision tree. A single
decision tree consists of *nodes* and *leaves* where the nodes reflect
the observations that are left after or before any split. The node is
split based on a splitrule. Usually the goal is to maximize reduction of
impurity in the resulting leaf (-nodes). This could be variance, shannon
entropy or gini index. For example, we might have an outcome such as
income and a predictor such as age. A split could be age  ≤ 21.

The terminal leaves are exclusive based on the predictors and reflect
the final decision. They consist of 1 or more observations (to predict,
we can fit e.g. a gaussian model or just use the median or mean for
regression and use the mode for classification). The number of samples
in the end leaves can be specified as a hyperparameter (see tuning RF).

The problem with a single decision tree would be that the model has a
has high variance. RF overcomes this by using an ensemble of trees
rather than a single tree at the cost of easy interpretability.
Furthermore, RF only selects a random fraction of the data
(**bagging**). Finally, RF considers only a subset of the *p* predictors
(the number of predictors considered is another hyperparameter of RF).
This leads to decorrelation of trees and less overfitting.

**Bagging:** Take random samples (with replacement) from the training
data set, fit a decision tree, and repeat *b* times. Then average
predictions of all trees. Since we average many trees, bagging improves
prediction accuracy (test accuracy) at the expense of interpretability.

Besides improved prediction accuracy, there is another advantage of
resampling: Since we only use about 2/3 of observations, there is a 1/3
*out-of-bag* sample, which can be used for error estimation (**out of
bag error**). With sufficiently large number of trees *b*, the OOB error
is virtually equivalent to the leave-one-out cross validation error (see
[here](http://appliedpredictivemodeling.com/blog/2014/11/27/vpuig01pqbklmi72b8lcl3ij5hj2qm)
for different types of cross-validation). This means: If we have very
low sample size, we should be able to use OOB without having to split
our data. Or: In case we split into train and test data, we not need an
extra validation set for hyperparameter tuning (see below).

The random forest algorithm is implemented (e.g.) in the *randomForest*
and *ranger* package. The ranger package is a faster implementation
(Wright & Ziegler, 2017). Both can be used by the caret package. The
latter offers many ML models in a structured manner.

Extremely randomized trees
--------------------------

The Extra-Trees (ET) algorithm differs by using the whole learning
sample rather than a bootstrap to grow the trees and by splitting the
nodes by choosing the cut-off points fully at random instead of
dependent on the outcome (Geurts, Ernst, & Wehenkel, 2006).  
It has two parameters: 1. *K*, the number of features randomly selected
at each node: By default we use $\\sqrt{p}$ rounded to the closest
integer for classification and *K* = *p* in regression. 2.
*n*<sub>*m**i**n*</sub>, the minimum sample size for splitting a node,
which by default is 2 for classification (fully grown trees) and 5 for
regression. *M*, (number of trees) could be considered an additional
parameter that should be high enough such that the variance converges.  
With regard to bias-variance-trade-off: The randomization of the
splitting point is meant to reduce the variance whereas using the full
sample is meant to decrease bias. This method can be used within the
*ranger* package by changing the default settings as described
[here](https://stats.stackexchange.com/questions/406666/what-exactly-is-the-extratrees-option-in-ranger/406680#406680).

Tuning RF
---------

In machine learning, for a model that you want to tune to improve the
accuracy, you need to split the data into training and testing data and
the training data into a training and a validation set. As mentioned,
one advantage of the random forest algorithm is the out-of-bag (OOB)
error. Remember that the algorithm selects always only a portion of the
data. The leftover portion is used as testing set to calculate the OOB
error. We can use OOB error to tune our model and do not need to
sacrifice any training data. In any case, it is necessary that you
evaluate the accuracy of the model using a test data set to avoid
overfitting or use a CV method when you have low *n*. In the following,
I mention the hyperparameters that are interesting for model tuning:

*mtry*  
The most important hyperparameter for tuning a random forest model is
*mtry* (the number of predictors considered at each split). If mtry ==
*p*, then RF == bagging. If mtry == 1, the split variable is completely
random since no longer gini index or entropy etc. could be used to make
a decision between predictors. When we have a large number of correlated
predictors, it is usually beneficial to use a smaller values of *m*, the
number of predictors considered for each split.

*ntree*  
We want to specify the number of trees as such that the out of bag error
converged. In general, this simply is a trade off between computational
requirements and accuracy where the accuracy at some point converges
with high enough ntree.

*sampsize*  
The default value for bagging is 0.6325. Reducing might induce bias,
increasing might induce overfitting.

*nodesize*  
This refers to the minimum number of samples in the terminal nodes.
Lower values mean more complex trees, which might lead to overfitting,
especially in noisy data. Higher values might cause more bias. This is
what above is referred to as *n*<sub>*m**i**n*</sub> under ExtraTrees
and usually is by default 2 for classification and 5 for regression
problems.

*maxnodes*  
The maximum number of terminal nodes. A low number limits the complexity
of trees.

Variable Importance
-------------------

Single trees are easy to interpret but have the disadvantage that they
suffer from high variance. In contrast, RF or ET loose easy
interpretability. To interpret tree ensembles we need to make use of the
calculated **variable importance**: There are two main importance
measures: **mean decrease of impurity (MDI)**, which is summing total
impurity reductions at all tree nodes where the variable appears and:
**mean decrease of accuracy (MDA)**, which is measuring accuracy
reduction on out-of-bag samples when the values of the predictor
variable are randomly permuted. The MDI is faster to compute, does not
require to use bootstrap sampling and empirically, it correlates well
with the MDA (except in specific conditions). In the following I explain
findings of (<span class="citeproc-not-found"
data-reference-id="louppeUnderstandingVariableImportances">**???**</span>)
(see article for terminology of relevant and irrelevant features):

If *K* = 1 (totally random trees) and we have predictors *X*<sub>1</sub>
and an almost identical predictor *X*<sub>2</sub> (identical means they
contain the same information but *X*<sub>2</sub> is a bit more noisy,
then the importance measure can show accurately the difference in
importance (if we have infinite samples). However, if K &gt; 1, then
these would be put into competition and there would be a masking effect
→ there can be relevant features with zero importances, in this case
*X*<sub>2</sub> would have zero importance. The importance of relevant
variables can be influenced by the number of irrelevant variables. E.g.
if *K* = 2 and there are irrelevant features *X*<sub>3</sub> …, then
there is the chance to draw *X*<sub>2</sub> and *X*<sub>3</sub> at the
top node and thus the importance of *X*<sub>2</sub> would be &gt; 0
again. Nevertheless, you still would find the strongly relevant features
in such a setting. To avoid false positives, one should use pruned trees
and non totally random trees (K &gt; 1), to avoid splits on irrelevant
features at the top nodes. Decreasing *D* or increasing *K* will however
increase the number of false negatives and affects the final ranking of
importance). To find all relevant features, choose *K* = 1. To find only
strongly relevant features: choose *K* = *p*. Thus, do not trust that a
higher importance really means the importance is higher. To make most of
the variable importance statistic do not focus on predictive
performance, which might lead to masking effects.Finally, the
instability of the importance score is higher than the instability of
predictive accuracy and you need higher number of trees for convergence.

Examples
--------

### Example 1: RF classification

I will use the childcare dataset to show an example model for
classification at first. In the following, I check whether we can
classify if a child entered childcare based on the abundance for 130
genera after entrance. A first useful step is to only select those
features that can be informative. For example, genera, which are only
abundant in less than 20% of the individuals might not help to predict
accurately.

``` r
# load packages
library(tidyverse)
library(microbiome)
library(randomForest)
library(ranger)
library(caret)
library(here)

# load helper functions
source("https://raw.githubusercontent.com/HenrikEckermann/in_use/master/mb_helper.R")

# loading this data file results in the follosing objects:
# - pseq.clr contains clr transformed abundances
# - pseq.rel contains relative abundances
# - df_imp: a list of dataframes with metadata (we only use one of those)
load(here("rdata/data_mi.rds"))

# we use rel ab for the selection of features
core_genus <- core_members(
  pseq.rel, 
  detection = 0.1/100, 
  prevalence = 20/100) %>% 
  clean_otu_names()

# how many features remain?
length(core_genus)
```

    ## [1] 64

``` r
df <- data_imp[[1]] %>% select(
  "sample_id", 
  "time",
  "cc", 
  core_genus)


# make sure our outcome is a factor:
df$cc %>% class()
```

    ## [1] "factor"

``` r
# it makes only sense that the model can differentiate after 
# infants have been in CC
df_post <- df %>% 
  filter(time == "post") %>%
  select(-sample_id, -time)


# fit default model but with higher number of trees
fit1 <- randomForest(
  formula = cc ~ .,
  ntree = 1e4,
  importance = TRUE,
  data = df_post
)

# As we can see the model performs poor:
fit1
```

    ## 
    ## Call:
    ##  randomForest(formula = cc ~ ., data = df_post, ntree = 10000,      importance = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 10000
    ## No. of variables tried at each split: 8
    ## 
    ##         OOB estimate of  error rate: 50%
    ## Confusion matrix:
    ##     no yes class.error
    ## no  27  22   0.4489796
    ## yes 27  22   0.5510204

``` r
# to access the estimated OOB:
fit1$err.rate[, 1] %>% median()
```

    ## [1] 0.5102041

``` r
# we could test if our performance metric is more extreme than 
# expected by chance but in this case this does not make sense
#rfUtilities::rf.significance(x = fit1, xdata = df_post, ntree = 25001, nperm = 1000)

# if our models would have turned out to perform well, we could
# check the most important features for classification:
fit1$importance %>% as.data.frame() %>%
  rownames_to_column("genus") %>% 
  arrange(MeanDecreaseGini) %>%
  mutate(genus = factor(genus, levels = genus)) %>%
  tail(10) %>%
  ggplot(aes(x = genus, y = MeanDecreaseGini)) +
  geom_bar(stat = "identity") +
  coord_flip() 
```

![](rf_tutorial_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r
# the OOB should perform well. Still we will look at leave one out
# crossvalidation using caret. You can specify "oob" to get OOB.
# If you want to use ranger, use "ranger" instead of "rf".
t_control <- trainControl(method = "CV")
fit2 <- train(
  form = cc ~ .,
  data = df_post,
  trControl = t_control,
  method = "rf",
  ntree = 1e4,
  tuneGrid = data.frame(mtry = 25)
)

# if we do not specify tuneGrid, it would try three different mtry
fit2$results
```

    ##   mtry  Accuracy Kappa AccuracySD   KappaSD
    ## 1   25 0.5366667 0.084  0.2837007 0.5542863

``` r
# check variable importance using caret. 
v_imp <- varImp(fit2, scale = F, useModel = T)
v_imp$importance %>% rownames_to_column("genus") %>% 
  arrange(Overall) %>%
  mutate(genus = factor(genus, levels = genus)) %>%
  tail(10) %>%
  ggplot(aes(x = genus, y = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip()
```

![](rf_tutorial_files/figure-markdown_github/unnamed-chunk-2-2.png)

``` r
df_post %>% dim()
```

    ## [1] 98 65

``` r
hyper_grid <- expand.grid(
  .mtry = c(1, 6, 8, 15, 45, 64),
  .splitrule = "extratrees",
  .min.node.size = 5)
  
fit_control <- trainControl(## 5-fold CV
                           method = "repeatedcv",
                           number = 5,
                           ## repeated 5 times
                           repeats = 5)

fit3_cc <- train(
  form = cc ~ .,
  data = df_post,
  method = "ranger",
  tuneGrid = hyper_grid,
  num.trees = 1e4,
  replace = FALSE,
  sample.fraction = 1,
  trControl = fit_control,
  importance = 'impurity'
)

fit3_cc
```

    ## Random Forest 
    ## 
    ## 98 samples
    ## 64 predictors
    ##  2 classes: 'no', 'yes' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 5 times) 
    ## Summary of sample sizes: 79, 79, 78, 78, 78, 78, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa       
    ##    1    0.5285263   0.056760040
    ##    6    0.4981053  -0.005116037
    ##    8    0.4941053  -0.013824015
    ##   15    0.5022105   0.002847372
    ##   45    0.4656842  -0.069947547
    ##   64    0.4676842  -0.066329193
    ## 
    ## Tuning parameter 'splitrule' was held constant at a value of
    ##  extratrees
    ## Tuning parameter 'min.node.size' was held constant at a
    ##  value of 5
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were mtry = 1, splitrule =
    ##  extratrees and min.node.size = 5.

Example 2: RF regression
------------------------

In the following example, I show RF regression where the outcome is a
measure of executive functioning and the predictors are again relative
abundances. For tuning we can choose to use the *tuneRF* function or the
*train* function from caret. We can also create or own grid with
hyperparameters through which we then iterate.

``` r
load(file = here("rdata/data.rds"))
# we already select the core members ()
core_genus2 %>% length()
```

    ## [1] 69

``` r
# we have several time points where bacterial abundance has been
# obtained and the meta variables are in a list for each timepoint
meta <- data_by_time[[1]]
meta_col <- colnames(meta)
data <- df_clr %>% 
  filter(sample_id %in% meta$sample_id) %>%
  select(sample_id, metacognition8_t, core_genus2) %>%
  na.omit()
# this is the final dataset with 125 samples and 69 genera + outcome + sample_id 
data %>% dim()
```

    ## [1] 131  71

``` r
# I use the ranger function so I can access OOB MSE and 
# use the extremely randomized trees algorithm
y <- "metacognition8_t"
x_train_id <- sample(data$sample_id, length(data$sample_id)/2)
x_test_id <- data %>% filter(!sample_id %in% x_train_id) %>% .$sample_id
x_train <- data %>% filter(sample_id %in% x_train_id) %>% select(-sample_id)
x_test <- data %>% filter(sample_id %in% x_test_id) %>% select(-sample_id)
fit1 <- ranger(
        as.formula(glue("{y} ~ .")),
        data = x_train,
        splitrule = "extratrees",
        sample.fraction = 1,
        replace = FALSE,
        num.trees = 1e4,
        importance = 'impurity',
        write.forest = TRUE
        
    )
fit1
```

    ## Ranger result
    ## 
    ## Call:
    ##  ranger(as.formula(glue("{y} ~ .")), data = x_train, splitrule = "extratrees",      sample.fraction = 1, replace = FALSE, num.trees = 10000,      importance = "impurity", write.forest = TRUE) 
    ## 
    ## Type:                             Regression 
    ## Number of trees:                  10000 
    ## Sample size:                      65 
    ## Number of independent variables:  69 
    ## Mtry:                             8 
    ## Target node size:                 5 
    ## Variable importance mode:         impurity 
    ## Splitrule:                        extratrees 
    ## OOB prediction error (MSE):       NaN 
    ## R squared (OOB):                  NaN

``` r
data$metacognition8_t %>% range()
```

    ## [1] 23 65

``` r
# fit1$variable.importance 
# # oob MSE
# fit1$prediction.error
# data$total_lns %>% plot()
y_pred <- predict(fit1, data = x_train)$predictions
cor.test(x_train$metacognition8_t, y_pred)
```

    ## 
    ##  Pearson's product-moment correlation
    ## 
    ## data:  x_train$metacognition8_t and y_pred
    ## t = 48.63, df = 63, p-value < 2.2e-16
    ## alternative hypothesis: true correlation is not equal to 0
    ## 95 percent confidence interval:
    ##  0.9786053 0.9920414
    ## sample estimates:
    ##       cor 
    ## 0.9869405

``` r
fit2 <- train(
  form = as.formula(glue("{y} ~ .")),
  data = data %>% select(-sample_id),
  method = "ranger",
  tuneLength = 5
)

# we see that the lower mtry values do better and the 
# splitingrule "extratrees" seems always is better when mtry >x.
# furthermore, extratrees is by default quite good whereas for
# RF we would benefit from specifying mtry manually as low here
plot(fit2)
```

![](rf_tutorial_files/figure-markdown_github/unnamed-chunk-3-1.png)

``` r
fit2
```

    ## Random Forest 
    ## 
    ## 131 samples
    ##  69 predictor
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 131, 131, 131, 131, 131, 131, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  splitrule   RMSE       Rsquared    MAE     
    ##    2    variance     9.677110  0.01444519  7.712491
    ##    2    extratrees   9.660383  0.01732698  7.729887
    ##   18    variance     9.850083  0.01916144  7.865851
    ##   18    extratrees   9.846963  0.02576183  7.885175
    ##   35    variance     9.937547  0.02057443  7.949623
    ##   35    extratrees   9.903664  0.02848237  7.943511
    ##   52    variance     9.979391  0.02280772  7.972907
    ##   52    extratrees   9.936993  0.02764739  7.960384
    ##   69    variance    10.027739  0.02077769  8.019869
    ##   69    extratrees   9.954707  0.03095921  7.979826
    ## 
    ## Tuning parameter 'min.node.size' was held constant at a value of 5
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were mtry = 2, splitrule =
    ##  extratrees and min.node.size = 5.

``` r
# for regression K = n is default. But lets us explore the effect
# of K for the algorithm is described by Geurts et al
hyper_grid <- expand.grid(
  .mtry = c(1, 2, 8, 15, 68),
  .splitrule = "extratrees",
  .min.node.size = c(5, 8, 11, 14, 17, 20, 23, 26, 29, 32))


fit_control <- trainControl(## 5-fold CV
                           method = "repeatedcv",
                           number = 5,
                           ## repeated 5 times
                           repeats = 5)
#y <- "brief_total8_t"
fit3 <- train(
  form = as.formula(glue("{y} ~ .")),
  data = data %>% select(-sample_id),
  method = "ranger",
  tuneGrid = hyper_grid,
  num.trees = 1e4,
  replace = FALSE,
  sample.fraction = 1,
  importance = 'impurity'
  #trControl = fit_control
)
fit3$results %>% arrange(RMSE)
```

    ##    mtry  splitrule min.node.size     RMSE   Rsquared      MAE    RMSESD
    ## 1     1 extratrees            32 9.282129 0.02030686 7.536250 0.8045926
    ## 2     1 extratrees            29 9.290108 0.02005388 7.542376 0.8052455
    ## 3     1 extratrees            26 9.296146 0.01974987 7.544514 0.8036820
    ## 4     1 extratrees            23 9.303808 0.01926599 7.550288 0.8014390
    ## 5     1 extratrees            20 9.310536 0.01849231 7.553322 0.7990009
    ## 6     1 extratrees            17 9.316538 0.01827140 7.554746 0.7979583
    ## 7     1 extratrees            14 9.328861 0.01738603 7.562243 0.7970858
    ## 8     1 extratrees            11 9.341158 0.01645882 7.568378 0.7919619
    ## 9     1 extratrees             8 9.355104 0.01534615 7.572659 0.7892412
    ## 10    2 extratrees            32 9.362355 0.02080585 7.597284 0.7967359
    ## 11    2 extratrees            29 9.370982 0.02101549 7.601541 0.7960768
    ## 12    1 extratrees             5 9.376819 0.01385068 7.583943 0.7746621
    ## 13    2 extratrees            26 9.378391 0.02048789 7.603122 0.7937066
    ## 14    2 extratrees            23 9.390423 0.02035890 7.611069 0.7896165
    ## 15    2 extratrees            20 9.400648 0.01986388 7.613991 0.7900226
    ## 16    2 extratrees            17 9.410503 0.01870887 7.618719 0.7885971
    ## 17    2 extratrees            14 9.419400 0.01797863 7.622749 0.7847068
    ## 18    2 extratrees            11 9.433578 0.01711159 7.629001 0.7816793
    ## 19    2 extratrees             8 9.450198 0.01657099 7.639577 0.7745255
    ## 20    2 extratrees             5 9.473484 0.01459807 7.654121 0.7638867
    ## 21    8 extratrees            32 9.520975 0.02128742 7.711016 0.7825788
    ## 22    8 extratrees            29 9.534513 0.02107385 7.717421 0.7831608
    ## 23    8 extratrees            26 9.545288 0.02095344 7.723276 0.7805202
    ## 24    8 extratrees            23 9.555777 0.02070143 7.726497 0.7764712
    ## 25    8 extratrees            20 9.567126 0.01952929 7.730845 0.7743479
    ## 26   15 extratrees            32 9.574520 0.01981083 7.751944 0.7846801
    ## 27    8 extratrees            17 9.580338 0.01917618 7.739245 0.7712991
    ## 28   15 extratrees            29 9.587706 0.01976507 7.759070 0.7817143
    ## 29    8 extratrees            14 9.593830 0.01827783 7.744679 0.7645641
    ## 30   15 extratrees            26 9.603038 0.01997183 7.767522 0.7761667
    ## 31    8 extratrees            11 9.606132 0.01746439 7.754280 0.7646646
    ## 32   15 extratrees            23 9.612025 0.01914833 7.769807 0.7735599
    ## 33   15 extratrees            20 9.625035 0.01842053 7.774971 0.7706644
    ## 34    8 extratrees             8 9.626180 0.01701651 7.771516 0.7579528
    ## 35   15 extratrees            17 9.640876 0.01822081 7.786192 0.7673375
    ## 36    8 extratrees             5 9.643335 0.01588014 7.788584 0.7571360
    ## 37   15 extratrees            14 9.651576 0.01741417 7.796848 0.7630215
    ## 38   15 extratrees            11 9.667126 0.01711275 7.809224 0.7595064
    ## 39   68 extratrees            32 9.678840 0.01722765 7.833293 0.7978348
    ## 40   15 extratrees             8 9.689899 0.01652245 7.832218 0.7587668
    ## 41   68 extratrees            29 9.695133 0.01707140 7.841422 0.7964230
    ## 42   15 extratrees             5 9.702265 0.01547941 7.846345 0.7513482
    ## 43   68 extratrees            26 9.711704 0.01733021 7.852236 0.7945944
    ## 44   68 extratrees            23 9.732753 0.01721317 7.864611 0.7887242
    ## 45   68 extratrees            20 9.745989 0.01690199 7.871881 0.7805345
    ## 46   68 extratrees            17 9.764782 0.01611121 7.889382 0.7759300
    ## 47   68 extratrees            14 9.777942 0.01639299 7.900546 0.7784453
    ## 48   68 extratrees            11 9.803181 0.01627930 7.927176 0.7739998
    ## 49   68 extratrees             8 9.823274 0.01632620 7.946404 0.7715955
    ## 50   68 extratrees             5 9.825420 0.01584965 7.952828 0.7564951
    ##    RsquaredSD     MAESD
    ## 1  0.03668581 0.8047625
    ## 2  0.03509157 0.8068733
    ## 3  0.03611898 0.8078531
    ## 4  0.03509348 0.8068849
    ## 5  0.03381081 0.8074462
    ## 6  0.03384415 0.8071166
    ## 7  0.03156605 0.8103355
    ## 8  0.03019534 0.8066843
    ## 9  0.02675368 0.8075278
    ## 10 0.03384713 0.8088408
    ## 11 0.03450211 0.8097547
    ## 12 0.02321175 0.7971373
    ## 13 0.03478043 0.8089192
    ## 14 0.03371087 0.8081034
    ## 15 0.03315741 0.8108255
    ## 16 0.03056614 0.8123608
    ## 17 0.02998702 0.8117978
    ## 18 0.02875647 0.8112464
    ## 19 0.02844504 0.8032981
    ## 20 0.02334538 0.7956435
    ## 21 0.02874005 0.8145060
    ## 22 0.02791484 0.8160208
    ## 23 0.02816860 0.8161804
    ## 24 0.02894141 0.8168923
    ## 25 0.02718145 0.8152958
    ## 26 0.02505135 0.8175097
    ## 27 0.02698660 0.8142053
    ## 28 0.02544573 0.8170526
    ## 29 0.02597588 0.8075993
    ## 30 0.02529939 0.8162744
    ## 31 0.02587748 0.8048165
    ## 32 0.02507090 0.8160509
    ## 33 0.02497916 0.8166209
    ## 34 0.02497063 0.7945196
    ## 35 0.02461222 0.8118782
    ## 36 0.02335228 0.7890905
    ## 37 0.02486946 0.8072301
    ## 38 0.02473117 0.8004882
    ## 39 0.02598450 0.8353604
    ## 40 0.02373769 0.7961533
    ## 41 0.02577011 0.8382446
    ## 42 0.02189635 0.7838498
    ## 43 0.02652460 0.8380578
    ## 44 0.02668978 0.8350269
    ## 45 0.02648114 0.8304690
    ## 46 0.02556851 0.8242693
    ## 47 0.02681852 0.8237067
    ## 48 0.02652988 0.8187547
    ## 49 0.02583684 0.8133907
    ## 50 0.02524822 0.7949875

``` r
plot(fit3)
```

![](rf_tutorial_files/figure-markdown_github/unnamed-chunk-3-2.png)

``` r
#y <- "brief_total8_t"
fit3_2193 <- train(
  form = as.formula(glue("{y} ~ .")),
  data = data %>% select(-sample_id),
  method = "ranger",
  tuneGrid = hyper_grid,
  num.trees = 1e4,
  replace = FALSE,
  sample.fraction = 1,
  importance = 'impurity'
  #trControl = fit_control
)
fit3_2193$results %>% arrange(RMSE)
```

    ##    mtry  splitrule min.node.size      RMSE   Rsquared      MAE    RMSESD
    ## 1     1 extratrees            32  9.506045 0.02839211 7.632628 0.6744629
    ## 2     1 extratrees            29  9.514773 0.02804736 7.638352 0.6773981
    ## 3     1 extratrees            26  9.524587 0.02846600 7.645177 0.6809076
    ## 4     1 extratrees            23  9.536645 0.02907811 7.654208 0.6823858
    ## 5     1 extratrees            20  9.546088 0.02825032 7.659039 0.6860519
    ## 6     1 extratrees            17  9.558986 0.02721333 7.667958 0.6903594
    ## 7     1 extratrees            14  9.577396 0.02694214 7.681470 0.6924592
    ## 8     2 extratrees            32  9.590225 0.02945358 7.697262 0.6857019
    ## 9     1 extratrees            11  9.592460 0.02533643 7.690751 0.6976808
    ## 10    2 extratrees            29  9.601400 0.02981450 7.705381 0.6878045
    ## 11    1 extratrees             8  9.615508 0.02350034 7.708616 0.7090763
    ## 12    2 extratrees            26  9.616348 0.02956919 7.716761 0.6916135
    ## 13    2 extratrees            23  9.629866 0.02985817 7.724128 0.6945746
    ## 14    2 extratrees            20  9.644149 0.02939105 7.735229 0.6984086
    ## 15    1 extratrees             5  9.655424 0.02258334 7.743547 0.7148522
    ## 16    2 extratrees            17  9.662670 0.02922508 7.748681 0.7019927
    ## 17    2 extratrees            14  9.678591 0.02781465 7.758248 0.7090876
    ## 18    2 extratrees            11  9.703054 0.02773369 7.778903 0.7125156
    ## 19    2 extratrees             8  9.732941 0.02670638 7.804059 0.7229459
    ## 20    8 extratrees            32  9.755580 0.03092858 7.834802 0.7179606
    ## 21    2 extratrees             5  9.769749 0.02515111 7.840189 0.7286164
    ## 22    8 extratrees            29  9.772203 0.03062284 7.848371 0.7238843
    ## 23    8 extratrees            26  9.789834 0.03119238 7.861425 0.7274072
    ## 24   15 extratrees            32  9.810906 0.02965487 7.882617 0.7394998
    ## 25    8 extratrees            23  9.811815 0.03182980 7.879399 0.7355475
    ## 26   15 extratrees            29  9.832110 0.02990247 7.899618 0.7405678
    ## 27    8 extratrees            20  9.832501 0.03197572 7.894947 0.7406560
    ## 28    8 extratrees            17  9.849403 0.03073981 7.910550 0.7469562
    ## 29   15 extratrees            26  9.852935 0.03058606 7.915750 0.7498849
    ## 30   15 extratrees            23  9.873969 0.03051202 7.935699 0.7538197
    ## 31    8 extratrees            14  9.879061 0.03075871 7.934099 0.7479314
    ## 32   15 extratrees            20  9.892668 0.03026901 7.951119 0.7596049
    ## 33    8 extratrees            11  9.898948 0.02933709 7.949204 0.7580484
    ## 34   15 extratrees            17  9.914412 0.03002546 7.970608 0.7654381
    ## 35    8 extratrees             8  9.923402 0.02793614 7.971290 0.7578454
    ## 36   68 extratrees            32  9.930992 0.02725782 7.993115 0.7902711
    ## 37   15 extratrees            14  9.939771 0.02949801 7.988256 0.7684034
    ## 38   68 extratrees            29  9.953155 0.02778293 8.011709 0.7881396
    ## 39    8 extratrees             5  9.954250 0.02664259 8.003783 0.7583719
    ## 40   15 extratrees            11  9.960587 0.02844057 8.004816 0.7735193
    ## 41   68 extratrees            26  9.970456 0.02777606 8.028710 0.7952738
    ## 42   15 extratrees             8  9.985345 0.02704575 8.028627 0.7748305
    ## 43   68 extratrees            23  9.994656 0.02841564 8.049035 0.7979246
    ## 44   15 extratrees             5 10.006947 0.02655587 8.054859 0.7711425
    ## 45   68 extratrees            20 10.013596 0.02830284 8.065250 0.8066817
    ## 46   68 extratrees            17 10.036458 0.02830477 8.087014 0.8051065
    ## 47   68 extratrees            14 10.061119 0.02831545 8.109361 0.8124103
    ## 48   68 extratrees            11 10.078951 0.02788809 8.122111 0.8068663
    ## 49   68 extratrees             8 10.089210 0.02670481 8.138068 0.8110892
    ## 50   68 extratrees             5 10.114319 0.02710066 8.168972 0.8086441
    ##    RsquaredSD     MAESD
    ## 1  0.04485555 0.6506131
    ## 2  0.04350341 0.6524584
    ## 3  0.04396219 0.6555647
    ## 4  0.04406851 0.6601173
    ## 5  0.04284186 0.6636375
    ## 6  0.04127102 0.6669035
    ## 7  0.03964699 0.6711185
    ## 8  0.04114273 0.6598023
    ## 9  0.03801109 0.6769304
    ## 10 0.04175404 0.6627352
    ## 11 0.03399141 0.6879012
    ## 12 0.03973116 0.6653244
    ## 13 0.04061757 0.6680036
    ## 14 0.03934218 0.6730822
    ## 15 0.02998946 0.6987543
    ## 16 0.03810193 0.6769459
    ## 17 0.03532820 0.6828784
    ## 18 0.03584755 0.6889479
    ## 19 0.03382885 0.6991685
    ## 20 0.03771394 0.6830756
    ## 21 0.03097321 0.7012041
    ## 22 0.03602404 0.6866121
    ## 23 0.03560192 0.6904149
    ## 24 0.03647060 0.7015789
    ## 25 0.03554304 0.6966379
    ## 26 0.03593369 0.7029369
    ## 27 0.03559546 0.7014151
    ## 28 0.03327421 0.7036015
    ## 29 0.03561975 0.7078660
    ## 30 0.03437408 0.7087335
    ## 31 0.03309691 0.7042387
    ## 32 0.03356592 0.7142276
    ## 33 0.03163348 0.7127869
    ## 34 0.03313863 0.7191540
    ## 35 0.03035292 0.7130759
    ## 36 0.03531533 0.7423446
    ## 37 0.03222405 0.7210087
    ## 38 0.03474762 0.7397594
    ## 39 0.02920489 0.7103969
    ## 40 0.03132447 0.7254660
    ## 41 0.03476674 0.7435665
    ## 42 0.02946674 0.7224499
    ## 43 0.03501238 0.7458231
    ## 44 0.02880110 0.7206073
    ## 45 0.03396623 0.7555982
    ## 46 0.03431314 0.7522991
    ## 47 0.03316834 0.7605539
    ## 48 0.03325601 0.7580792
    ## 49 0.03183278 0.7547704
    ## 50 0.03140358 0.7561161

``` r
plot(fit3_2193)
```

![](rf_tutorial_files/figure-markdown_github/unnamed-chunk-3-3.png)

References
----------

Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized
trees. *Machine Learning*, *63*(1), 3–42.
<https://doi.org/10.1007/s10994-006-6226-1>

Wright, M. N., & Ziegler, A. (2017). Ranger: A Fast Implementation of
Random Forests for High Dimensional Data in C++ and R. *J. Stat. Soft.*,
*77*(1). <https://doi.org/10.18637/jss.v077.i01>
