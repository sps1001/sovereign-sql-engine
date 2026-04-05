import sqlitecloud
import os
from dotenv import load_dotenv
load_dotenv()

# Load values from environment
host = os.getenv("SQLITE_HOST")
port = os.getenv("SQLITE_PORT")
db = os.getenv("SQLITE_DB")
api_key = os.getenv("SQLITE_API_KEY")


# Build connection string
conn_str = f"sqlitecloud://{host}:{port}/{db}?apikey={api_key}"

# Open the connection to SQLite Cloud
conn = sqlitecloud.connect(conn_str)
cursor = conn.execute("SELECT * FROM table_metadata")
# cursor = conn.execute("PRAGMA database_list;")
result = cursor.fetchone()

print(result)

conn.close()
