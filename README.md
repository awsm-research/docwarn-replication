A replication package of "Towards Reliable Agile Iterative Planning via Predicting Documentation Changes of Work Items", published at the 19th International Conference on Mining Software Repositories – Technical track (MSR 2022 Technical track).

------------------------------------------------------

DocWarn-T and DocWarn-H code can be found in /code/model/DocWarn_H.py and DocWarn_T.py
DocWarn-C code can be found in /code/Rscript/DocWarn_C.R

The analysis code for RQ1 can be found in /code/Rscript.
RQ1_performance_measure.R must be run first to measure the performance of the model.
Then, run RQ1_performance_stattest.R to perform statistical test on the measured performance.

RQ3_rank_features.R is used to find a statistical distinct rank for each features in DocWarn-C.


Results and data are located in /data
/data/data_reverted_cleaned stores dataset that the work items were reverted to sprint assignment time.
/data/trainingData stores the dataset for each cross-validation round.
/data/features stores the metrics extracted from each work items in the dataset.
/data/modelResult stores the DocWarn-C R models (/models/...), performance of each DocWarn variations, and the result of features ranking.

/distilroberta-base-jira is the fine-tuned version of distilroberta-base with 110k JIRA issues. (will be provided after review completed due to the total files size)

Noted that the DocWarn-C R models and fine-tuned distilroberta are only available on Figshare: https://figshare.com/s/88547b3c197b21b60f7c 


rq2_manual_validation is the result of RQ2's manual classification to validate DocWarn-C.
rq2_manual_validation_external is the manual classification that were done by the external coder (to measure the inter-rater agreement).
code 0 = others, 1 = changing scope, 2 = defining scope, 3 = adding additional detail, 4 = adding implementation detail