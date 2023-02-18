# python imports
import argparse
# pypi imports
from pymongo import MongoClient
import datetime
import pandas as pd



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog = 'MED/MSC Mozzware Recording Extractor',
                        description = 'Copy recordings for a UUID list in a date range into a working directory',
                        )
    parser.add_argument("uuids", help="CSV file containing UUIDs.  Format is 1 column with header")
    parser.add_argument("dates", help="CSV file containing recordings dates. Format is 1 column with header and YYYY-MM-DD formated dates")
    parser.add_argument("mozzDir", help="Mount point for MozzWear Recordings (e.g. /dbmount/MozzWear)")
    args = parser.parse_args()

    uuid_file = args.uuids
    dates_file = args.dates
    # start_datetime = datetime.datetime.strptime(args.start, r"%Y-%m-%d")
    # end_datetime = datetime.datetime.strptime(args.end, r"%Y-%m-%d") + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)

    uuid_list = pd.read_csv(uuid_file).iloc[:,0:1]
    dates_list = pd.read_csv(dates_file).iloc[:,0:1]
    date_column = dates_list.columns[0]
    dates_list.sort_values(date_column, inplace=True)

    start_of_dates = dates_list[date_column].apply(lambda x: datetime.datetime.strptime(x, r"%Y-%m-%d"))
    end_of_dates = dates_list[date_column].apply(lambda x: datetime.datetime.strptime(x, r"%Y-%m-%d") + datetime.timedelta(days=1) - datetime.timedelta(seconds=1))

    # connect to database
    client = MongoClient('mongodb://humbug.ac.uk/')
    db = client['backend_upload']
    recordings = db['reports']

    recordings_df_list = []
    for start_datetime, end_datetime in zip(start_of_dates, end_of_dates):
        myquery = {"uuid": {"$in": uuid_list}, "datetime_recorded": {"$gt": start_datetime, "$lt": end_datetime}}
        mydoc = recordings.find(myquery)

        recording_df = pd.DataFrame(list(mydoc))

        recording_df["current_path"] = recording_df['path'].str.replace('/data/MozzWear', args.mozzDir)

        recordings_df_list.append(recording_df)

    pd.concat(recordings_df_list, ignore_index=True)\
        .to_csv(f"MED_Task_{dates_list.iloc[0][date_column]}_to_{dates_list.iloc[-1][date_column]}.csv", index=False)








