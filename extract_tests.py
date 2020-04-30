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
import os

from convert_eprime.convert import _text_to_df
import pandas as pd


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_file',
                   help='BIDS folder to convert.')
    p.add_argument('in_task',
                   help='task you want to convert.')
    p.add_argument('out_file',
                   help='output csv file.')
    return p


def assert_task_exists(parser, in_task):
    fullpath = os.path.join('configs', in_task + '.json')
    if os.path.exists(fullpath):
        with open(fullpath, 'r') as json_file:
            task = json.load(json_file)
    else:
        parser.error('{} does not exist'.format(in_task))

    return task


def assert_task_df(df, task):
    columns = []
    for nDesc in task["descriptions"]:
        if not set([nDesc["name"]]).issubset(df.columns):
            logging.error("{} does not exist into df".format(nDesc["name"]))


def get_ioi(df, ioi_dict):
    """
    Get Index of Interest from task['ioi'] list of dict.

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
    index = np.zeros(df_len, ), dtype=bool)
    for curr_condition in ioi_dict:
        columnName = curr_condition["name"]
        if isinstance(curr_condition["value"], list):
            for columnValue in curr_condition["value"]:
                curr_index = df[columnName]==columnValue
                index = index | curr_index
        else:
            curr_index = df[columnName]==curr_condition["value"]
            index = index | curr_index

    return indexes


def extract_specific(df, task):
    assert_task_df(df, task)

    #  Get usefull indexes: ioi -> index of interest
    ioi = get_ioi(df, task['ioi'])

    TTL = float(df.loc[0, task['TTL']])
    subDF = df[ioi].copy()
    selected_columns = []
    for nDesc in task["descriptions"]:
        selected_columns.append(nDesc["name"])
        if nDesc["type"] == "rt":
            subDF[nDesc["name"]] = subDF[nDesc["name"]].astype(float).apply(_subTTL, var=TTL)
        elif nDesc["type"] == "time":
            subDF[nDesc["name"]] = subDF[nDesc["name"]].astype(float).apply(_divide)


    subDF = subDF[selected_columns].copy()
    df_renamed = rename_df(subDF, task)
    return df_renamed

def _divide(val):
    return val/1000

def _subTTL(val, var):
    return (val - var) / 1000

def rename_df(df, task):
    for nDesc in task["descriptions"]:
        df = df.rename(columns={nDesc["name"]: nDesc["new_name"]})

    return df

def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    in_file = args.in_file
    in_task = args.in_task
    out_file = args.out_file

    # Get json file from task name
    task = assert_task_exists(parser, in_task)

    #  Convert eprime txt file to data frame
    df = _text_to_df(in_file)
    #new_df = extract_specific(df, task)
    df.to_csv(out_file, index=False, sep='\t')


if __name__ == "__main__":
    main()
