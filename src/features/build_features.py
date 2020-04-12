import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler


def main():
    """ Runs feature building and places data in ../processed
    """
    logger = logging.getLogger(__name__)
    logger.info('Building features')

    raw_df = pd.read_csv('../../data/raw/creditcard.csv')
    rob_scaler = RobustScaler()

    raw_df['scaled_amount'] = rob_scaler.fit_transform(raw_df['Amount'].values.reshape(-1, 1))
    raw_df['scaled_time'] = rob_scaler.fit_transform(raw_df['Time'].values.reshape(-1, 1))
    raw_df.drop(['Time', 'Amount'], axis=1, inplace=True)

    raw_df.sample(frac=1, replace=False, random_state=0, axis=0)

    raw_df.to_csv(f'../../data/processed/processed.csv')

    logger.info(f'Features built. \n Features: {raw_df.columns}')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)


    main()