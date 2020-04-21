import pandas as pd
import logging

from numpy import mean

from constants import (
    baseline_classifiers,
    LogisiticRegression_grid,
    model_metrics,
    best_model_file_name,
    LogisticRegression_rndm_params
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.externals import joblib


def main():
    """ Trains a logistic regression, an attempt to be 'production' grade
    """

    logger = logging.getLogger(__name__)
    logger.info(f'Reading data')
    processed_df = pd.read_csv('../../data/processed/processed.csv')

    X = processed_df.drop('Class', axis=1).values
    y = processed_df['Class'].values

    accuracy_lst = []
    precision_lst = []
    recall_lst = []
    f1_lst = []

    rand_log_reg = RandomizedSearchCV(
        baseline_classifiers['LogisticRegression'],
        LogisticRegression_rndm_params,
        n_iter=4
    )

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    logger.info(f'Constructing model pipeline and cross validating')
    idx=1
    for train, test in skf.split(X=X, y=y):
        logger.info(f'Run {idx}')
        model = Pipeline(
            [
                ('sampling', SMOTE(sampling_strategy='minority')),
                ('classification', rand_log_reg)
            ]
        )

        model.fit(X[train], y[train])
        best_estimators = rand_log_reg.best_estimator_
        prediction = best_estimators.predict(X[test])

        accuracy_lst.append(model.score(X[test], y[test]))
        precision_lst.append(precision_score(y[test], prediction))
        recall_lst.append(recall_score(y[test], prediction))
        f1_lst.append(f1_score(y[test], prediction))
        idx+=1

    metrics = f'''
    Accuracy: {mean(accuracy_lst)} \n
    Precision: {mean(precision_lst)} \n
    Recall: {mean(recall_lst)} \n
    F1: {mean(f1_lst)}
          '''

    print(metrics)

    f = open(f'../../models/metrics.txt', 'w')
    f.write(metrics)
    f.close()

    joblib.dump(rand_log_reg, f'../../models/{best_model_file_name}', compress=9)
    logger.info(f'Serialised model as {best_model_file_name}')

    return rand_log_reg


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
