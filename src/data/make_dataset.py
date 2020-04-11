# -*- coding: utf-8 -*-
import click
import logging
import kaggle
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.option('-data', prompt='Dataset')
def main(data):
    """ Runs data query and places data in ../raw
    """
    logger = logging.getLogger(__name__)
    logger.info('Querying dataset and unzipping')
    kaggle.api.dataset_download_files(data, path='../../data/raw/', unzip=True)
    logger.info('Data query complete.')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
