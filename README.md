# A replication package of "Towards Reliable Agile Iterative Planning via Predicting Documentation Changes of Work Items"
#### published at the 19th International Conference on Mining Software Repositories – Technical track (MSR 2022 Technical track).
------------------------------------------------------

### Replication code
- DocWarn-T: /code/model/DocWarn_T.py
- DocWarn-H: /code/model/DocWarn_H.py
- DocWarn-C: /code/Rscript/DocWarn_C.R

### Analysis code for RQ1
- can be found at /code/Rscript/
- RQ1_performance_measure.R must be run first to measure the performance of the model.
- Then, run RQ1_performance_stattest.R to perform statistical test on the measured performance.
- RQ3_rank_features.R is used to find a statistical distinct rank for each features in DocWarn-C.

### Results and dataset
- can be found at /data (only available on on Figshare version: https://figshare.com/s/88547b3c197b21b60f7c)
- /data/data_reverted_cleaned stores dataset that the work items were reverted to sprint assignment time.
- /data/trainingData stores the dataset for each cross-validation round.
- /data/features stores the metrics extracted from each work items in the dataset.
- /data/modelResult stores the DocWarn-C R models (/models/...), performance of each DocWarn variations, and the result of features ranking.

### Manual classification
- /rq2_manual_validation.csv is the result of RQ2's manual classification to validate DocWarn-C.
- /rq2_manual_validation_external.csv is the manual classification that were done by the external coder (to measure the inter-rater agreement).
- code 0 = others, 1 = changing scope, 2 = defining scope, 3 = adding additional detail, 4 = adding implementation detail

### DistilRoberta for DocWarn-T and DocWarn-H
- can be found at /distilroberta-base-jira  (only available on on Figshare version: https://figshare.com/s/88547b3c197b21b60f7c)
- This is the fine-tuned version of distilroberta-base with 110k JIRA issues. 



