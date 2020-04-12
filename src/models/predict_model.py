from sklearn.externals import joblib
from constants import best_model_file_name
import logging

import pandas as pd

def main():
    """ Performs inference/predictions on test set
    """
    logger = logging.getLogger(__name__)

    logging.info(f'Loading Model')
    model = joblib.load(best_model_file_name)

    logging.info(f'Loading unseen data')
    test_df = pd.read_csv('test.csv')
    X_test = test_df.values

    logging.info(f'Outputting prediction probabilities')

    y_hat = model.predict_proba(X_test)

    return y_hat


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()