#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Convert txt data from eprime to tsv using convert_eprime.
    git@github.com:tsalo/convert-eprime.git

    HCPTRT tasks
    https://github.com/hbp-brain-charting/public_protocols

"""

import argparse
import json
import logging
import numpy as np
import os

from convert_eprime.convert import _text_to_df
import pandas as pd


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_file',
                   help='BIDS folder to convert.')
    p.add_argument('in_task',
                   help='task you want to convert (wm, emotion, gambling...).')
    p.add_argument('out_file',
                   help='output tsv file.')
    p.add_argument('-v', action='store_true', dest='verbose',
                   help='If set, produces verbose output.')
    return p


def assert_task_exists(parser, in_task):
    fullpath = os.path.join('configs', in_task + '.json')
    if os.path.exists(fullpath):
        with open(fullpath, 'r') as json_file:
            task = json.load(json_file)
    else:
        parser.error('{} does not exist'.format(in_task))

    return task


def assert_task_df(df_columns, json_dict):
    """
    Assert if json_dict is using column name that exists in df_columns.
    Does not look into ioi
    Parameters
    ----------
    df_columns : int
        length of the pandas dataframe.

    json_dict : dict
        Dict to select rows of interest.

    Returns
    -------
    """

    allColumns = []

    for curr_event in json_dict["events"]:
        for curr_key in curr_event:
            if "column" in curr_key:
                if isinstance(curr_key["column"], list):
                    allColumns = allColumns + curr_key["column"]
                else:
                    allColumns.append(curr_key["column"])

            if isinstance(curr_key, list):
                for listKey in curr_key:
                    if isinstance(curr_key["column"], list):
                        allColumns = allColumns + curr_key["column"]
                    else:
                        allColumns.append(curr_key["column"])

    for curr_column in allColumns:
        if not set([curr_column]).issubset(df.columns):
            logging.error("{} does not exist into df".format(curr_column))


def get_ioi(df, ioi_dict):
    """
    Get Index of Interest from event dict.

    Parameters
    ----------
    df_len : int
        length of the pandas dataframe.

    ioi_dicts : list of dict
        Dict to select rows of interest.

    Returns
    -------
    ioi: pandas.core.series.Series
        Series of Index of interest.
    """
    indexes = np.zeros((len(df), ), dtype=bool)
    for curr_condition in ioi_dict:
        columnName = curr_condition["column"]
        if isinstance(curr_condition["value"], list):
            for columnValue in curr_condition["value"]:
                curr_index = df[columnName] == columnValue
                indexes = indexes | curr_index
        elif curr_condition["value"] == 'notNAN':
            curr_index = ~np.isnan(df[columnName].astype(np.float))
            indexes = indexes | curr_index
        else:
            curr_index = df[columnName] == curr_condition["value"]
            indexes = indexes | curr_index

    return indexes


def get_onsets(df, onset_dict):
    """
    Get onsets from event dict.

    Parameters
    ----------
    df : pandas DataFrame
        current DataFrame

    onset_dict : dict
        Dict to select rows of interest for onset.

    Returns
    -------
    onset_serie: pandas.core.series.Series
        Series of Index with onsets.
    """

    if 'value' in onset_dict:
        if onset_dict['value'] == 'merge' and isinstance(onset_dict['column'],
                                                         list):
            onset_serie = merge_columns(df, onset_dict['column'])
        elif isinstance(onset_dict['column'], list):
            logging.error('Use case {} not coded'.format(onset_dict['value']))
        else:
            logging.error('If you want to merge you need at least two columns')
    else:
        onset_serie = merge_columns(df, [onset_dict['column']])

    onset_serie = onset_serie.rename('onset')

    return onset_serie


def get_durations(df, onset, duration_dict):
    """
    Get durations from event dict.

    Parameters
    ----------
    df : pandas DataFrame
        current DataFrame

    duration_dict : dict
        Dict to select rows or compute duration.

    Returns
    -------
    duration_serie: pandas.core.series.Series
        Series of Index with durations.
    """
    if 'formula' in duration_dict:
        curr_formula = duration_dict['formula']
        if len(curr_formula) == 3:
            duration_serie = df[duration_dict['column']]
            duration_serie = duration_serie.shift(periods=curr_formula[0])
            if curr_formula[1] == 'subtract':
                duration_serie = duration_serie.astype(np.float) - onset.astype(np.float)
            elif curr_formula[1] == 'add':
                duration_serie = duration_serie.astype(np.float) + onset.astype(np.float)
        else:
            logging.error('This is not coded yet')
    elif isinstance(duration_dict['column'], str):
        duration_serie = df[duration_dict['column']]

    duration_serie = duration_serie.rename('duration')
    return duration_serie


def merge_columns(df, listColumns):
    """
    Merge column inot Get Index of Interest from task['ioi'] list of dict.

    Parameters
    ----------
    df : pandas DataFrame
        Current DataFrame.

    listColumns : list
        List of column names to merge. You should not have overlap between
        these columns.

    Returns
    -------
    new_serie: pandas.core.series.Series
        Merge serie.
    """
    new_serie = pd.Series([])
    for curr_column in listColumns:
        new_serie = new_serie.combine_first(df[curr_column])

    return new_serie


def get_key(df, columnName, key_dict):
    """
    Get Key from event dict.

    Parameters
    ----------
    df : pandas DataFrame
        Current DataFrame.

    columnName: str
        Rename column with this name.

    key_dict : dict
        Dict to select rows or compute duration.

    Returns
    -------
    key_serie: pandas.core.DataFrame
        Series from specific key.
    """

    if isinstance(key_dict['column'], str):
        key_serie = df[key_dict['column']]
        if key_dict['type'] == 'stim':
            key_serie = '../../stimulis/' + key_serie
        elif columnName == "nbloc":
            key_serie = key_serie.astype(np.float) - \
                            np.floor(key_serie.astype(np.float)/2)

    elif isinstance(key_dict['column'], list):
        if key_dict['value'] == 'merge':
            key_serie = merge_columns(df, key_dict['column'])
        else:
            logging.error('not coded yet')
    else:
        logging.error('not coded yet')

    key_serie = key_serie.rename(columnName)

    return key_serie


def extract_event(df, curr_event, new_df, ttl):

    #  Get usefull indexes: ioi -> index of interest
    ioi = get_ioi(df, curr_event['ioi'])

    # Get onsets and durations
    onsets = get_onsets(df, curr_event['onset'])
    durations = get_durations(df, onsets, curr_event['duration'])

    # Creation of the event dataframe
    listOfType = [curr_event['name']] * len(ioi)
    curr_event_df = pd.DataFrame(np.asarray(listOfType), columns=['type'])

    # Delete already used keys
    del curr_event['onset']
    del curr_event['duration']
    del curr_event['name']
    del curr_event['ioi']

    # Loop over all keys
    for curr_key in curr_event.keys():
        curr_event_df = curr_event_df.join(get_key(df, curr_key,
                                                   curr_event[curr_key]))

    # Reduce all df to index of interest
    curr_event_df = curr_event_df[ioi]
    onsets = onsets[ioi]
    durations = durations[ioi]

    # Cast into np.float and divide by 1000
    onsets = onsets.astype(np.float).apply(_subTTL, var=ttl)
    durations = durations.astype(np.float).apply(_divide)

    # Join all df into one
    curr_event_df = curr_event_df.join(onsets)
    curr_event_df = curr_event_df.join(durations)

    # Concat each event_df into one
    new_df = pd.concat([new_df, curr_event_df])

    return new_df


def get_TTL(df, column):
    return float(df.loc[0, column])


def _divide(val):
    return val/1000


def _subTTL(val, var):
    return (val - var) / 1000


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    in_file = args.in_file
    in_task = args.in_task
    out_file = args.out_file

    # Get json file from task name
    task = assert_task_exists(parser, in_task)

    #  Convert eprime txt file to data frame
    df = _text_to_df(in_file)

    # Assert all columns in task exist in df
    assert_task_df(df.columns, task)

    # Final df
    new_df = pd.DataFrame()

    # Get TTL
    TTL = get_TTL(df, task['TTL'])

    for curr_event in task["events"]:
        logging.info('Current event: {}'.format(curr_event['name']))
        new_df = extract_event(df, curr_event, new_df, TTL)

    # Sort new_df by onset
    new_df = new_df.sort_values('onset')
    logging.info(new_df)
    # Extract new_df into tsv file
    new_df.to_csv(out_file, index=False, sep='\t')


if __name__ == "__main__":
    main()
