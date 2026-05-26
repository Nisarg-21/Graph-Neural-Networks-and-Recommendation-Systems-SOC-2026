# part 2

## phase 2

### correlation explanation

Correlation is a measure of how strongly two variables are related to each other, with values ranging from -1 to 1. A value close to 1 means both variables move in the same direction, close to -1 means they move in opposite directions, and close to 0 means there is no meaningful relationship between them.

### why multi-collinearity can introduce redundancy into a model.

Looking at the heatmap, clear multicollinearity can be spotted between several features. For example x_05 and x_02 show a correlation of 0.78 with each other, and x_25 and x_100 show 0.74. This means these pairs of features are carrying almost the same information. When both features from such a pair are fed into a model together, the model does not gain anything new from the second one ,it just sees the same pattern twice. This makes the model unstable and adds redundancy without improving its ability to predict late submissions.

## phase 3

### Which algorithm preserved the global structure of the data better?

UMAP preserved the global structure better. The clusters in the UMAP plot are more compact and well defined compared to t-SNE. t-SNE is good at keeping nearby points together but tends to distort the overall layout of the data.

### Do the late and on-time points form clean clusters, or are they deeply intermingled?

In all 3 plots the red (late) and blue dots are deeply mixed within every cluster. There is no single cluster that is purely one class. This confirms what the models also showed, that late submissions do not have a clearly distinct pattern that separates them from on time submissions.

### Compare the plots of 100 features with 15 features, which looks to have better clustering?

The 100 feature UMAP had fewer but tighter clusters compared to the 15 feature UMAP which was more scattered. More features means UMAP has more information to find similarities between points, so naturally it groups them better.

# part 3

### Deconstruct the Metrics: Based on the output of Scikit-Learn's classification_report, explain exactly what Accuracy, Precision, Recall, F1-Score and the AUC-ROC represent in the context of this specific problem. 

accuracy:out of all 324 test samples, how many did the model get right overall. 
precesion:out of all the submissions the model predicted as late ,how many were actually late.
recall :  out of all the submissions that were actually late, how many did the model correctly catch.A low recall for class 1 means the model is missing real late submissions
F1-Score:  F1 = 2 × (Precision × Recall) / (Precision + Recall),so F1 is the overall evaluation between precesion and recall.
AUC-ROC: it measures how well the model can distinguish between late and on time submissions across all possible thresholds.A value of 0.5 means the model is no better than random guessing and 1.0 means perfect.my models all scored around 0.5 which means they were barely better than a coin flip at separating the two classes.

### Model Selection: Which model performed the best overall? Why do you think that specific architecture succeeded over the others on this dataset?

SVC performed the best overall among the three models. It had the highest AUC-ROC of 0.5371, highest F1 score of 0.1978 and highest precision of 0.1268, making it the most reliable at identifying late submissions compared to Logistic Regression and Random Forest.SVC works well on this dataset because it tries to find the best boundary that separates late from on time submissions with the maximum gap possible. Unlike Random Forest which just went with the majority class, SVC with class_weight='balanced' was forced to pay equal attention to both classes. It also handles high dimensional data well which fits our case since we have 15 features.

### Practical Application: Imagine this model is deployed by an operations team at an organization or educational institution. Explain what practical purpose this model serves.

This model can be used by delivery companies like Zepto or Blinkit to predict which deliveries are likely to be late before they actually are. Once the model flags a potential delay, the company can immediately notify the customer that their delivery might be late instead of leaving them waiting without any information. 

### More Features?: As a test try implementing the models with all features and see if it gives you something better.
Using all 100 features gave a slight improvement in recall and F1 score for Logistic Regression and SVC compared to using just the top 15 features. However the improvement was not significant enough to justify the added complexity of using all 100 features

# part 4

### Explain how the frequency of oscillation in Case A would change if u significantly increased the total energy E further away from V.

The frequency of oscillation depends on k = √2(E-V). If E increases further away from V, the value of k gets larger, which means the cosine wave completes more cycles in the same space ,so the frequency increases and the wave oscillates faster.

### What physical meaning does a rapidly decaying wavefunction (case B) have regarding the probability of finding an electron inside a highly resistive potential barrier?

The rapidly decaying wavefunction means the probability of finding the electron gets smaller and smaller the deeper it goes into the barrier. Near the surface there is still some small chance of finding it but as you go further in that chance drops almost to zero.