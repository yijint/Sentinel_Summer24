import sqlalchemy
import sqlalchemy.orm
import pandas as pd

# Database credentials and settings
db_user = 'dataprep'
db_pass = 'sdK77:+,^^g[+rbV'
db_name = 'images'
db_ip   = '127.0.0.1:2235'  # Corrected IP address
cloud_sql_connection_name = 'sentinel-project-278421:us-east4:training-data'
TABLE_NAME = 'images_v3'
DATASET_NAME = 'nz-trailcams-test'
URL = f'mysql+pymysql://{db_user}:{db_pass}@{db_ip}/{db_name}'

# read dataset
def print_dataset():
    engine = sqlalchemy.create_engine(URL, pool_size=5, max_overflow=2, pool_timeout=30, pool_recycle=1800)
    openesc, closeesc = '', '' # escape column names
    query = f'SELECT * FROM images.{TABLE_NAME}'
    query += f' WHERE {openesc}dataset{closeesc} = "{DATASET_NAME}"'
    db_df = pd.read_sql(query, con=engine)
    return db_df

# delete data
def delete_dataset():
    engine = sqlalchemy.create_engine(URL, pool_size=5, max_overflow=2, pool_timeout=30, pool_recycle=1800)
    with engine.connect() as connection:
        openesc, closeesc = '', '' # escape column names
        query = f'DELETE FROM images.{TABLE_NAME}'
        query += f' WHERE {openesc}dataset{closeesc} = "{DATASET_NAME}"'
        connection.execute(sqlalchemy.text(query))

# test
with open("scratch.txt", 'w') as f:
    print(f"dataset:\n{print_dataset()}", file = f, flush=True)
    print("==========================================================", file = f, flush=True)
    try:
        delete_dataset()
        print("Dataeset deleted", file = f, flush=True)
    except Exception as ex:
        print(f"Deletion failed: {ex}", file = f, flush=True)
    print("==========================================================", file = f, flush=True)
    print(f"dataset:\n{print_dataset()}", file = f, flush=True)
9