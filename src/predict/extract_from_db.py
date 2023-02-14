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
    parser.add_argument("uuids", help="CSV file containing UUIDs.  Format should be 1 column with header")
    parser.add_argument("start", help="Start of date range YYYY-MM-DD")
    parser.add_argument("end", help="End of date range YYYY-MM-DD")
    parser.add_argument("mozzDir", help="Mount point for MozzWear Recordings (e.g. /dbmount/MozzWear)")
    parser.add_argument()
    args = parser.parse_args()

    uuid_file = args.uuids
    start_datetime = datetime.datetime.strptime(args.start, r"%Y-%m-%d")
    end_datetime = datetime.datetime.strptime(args.end, r"%Y-%m-%d") + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)

    uuid_list = pd.read_csv(uuid_file).iloc[:,0].tolist()

    # connect to database
    client = MongoClient('mongodb://humbug.ac.uk/')
    db = client['backend_upload']
    recordings = db['reports']

    myquery = {"uuid": {"$in": uuid_list}, "datetime_recorded": {"$gt": start_datetime, "$lt": end_datetime}}
    mydoc = recordings.find(myquery)

    recording_df = pd.DataFrame(list(mydoc))

    recording_df["current_path"] = recording_df['path'].str.replace('/data/MozzWear', args.mozzDir)

    recording_df.to_csv(f"MED_Task_{start_datetime.date()}_to_{end_datetime.date()}.csv", index=False)








