import pandas as pd
import logging

from constants import (
    random_seed,
    baseline_classifiers,
    LogisiticRegression_grid,
    model_metrics,
    best_model_file_name
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split as tts,
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

    X_train, X_test, y_train, y_test = tts(X, y, random_state=random_seed)

    logger.info(f'Constructing model pipeline')
    model = Pipeline(
        [
            ('sampling', SMOTE()),
            ('classification', baseline_classifiers['LogisiticRegression'])
        ]
    )

    logger.info(f'Constructing baseline model')
    model.fit(X_train, y_train)
    baseline_y_hat = model.predict(X_test)

    baseline_report = classification_report(y_test, baseline_y_hat)
    print(f'Classification report for Baseline model \n{baseline_report}')

    logger.info(f'Performing Gridsearch')
    gridsearch_cv = GridSearchCV(
        estimator=model,
        param_grid=LogisiticRegression_grid,
        cv=5,
        scoring=model_metrics,
        n_jobs=1,
        refit='F1',
        return_train_score=True
    )

    gridsearch_cv.fit(X_train, y_train)
    print(f'Best score (log-loss): {gridsearch_cv.best_score_}\nBest Parameters: {gridsearch_cv.best_params_}')
    gridsearch_y_hat = gridsearch_cv.predict(X_test)
    gridsearch_report = classification_report(y_test, gridsearch_y_hat)
    print(f'Classification report for tuned model \n{gridsearch_report}')

    joblib.dump(gridsearch_cv, best_model_file_name, compress=9)
    logger.info(f'Serialised model as {best_model_file_name}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
