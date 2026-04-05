import os
import csv
import io
import sys
import sqlitecloud
from dotenv import load_dotenv

load_dotenv()

# Configuration
conn_str = f"sqlitecloud://{os.getenv('SQLITE_HOST')}:{os.getenv('SQLITE_PORT')}/{os.getenv('SQLITE_DB')}?apikey={os.getenv('SQLITE_API_KEY')}"


def open_csv_with_fallback(file_path):
    with open(file_path, mode="rb") as source_file:
        raw_bytes = source_file.read()

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return io.StringIO(raw_bytes.decode(encoding, errors="strict"))
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Unable to decode CSV file with supported encodings: {file_path}")


def load_table_descriptions():
    descriptions = {}

    with open_csv_with_fallback(TABLE_METADATA_FILE) as metadata_file:
        reader = csv.DictReader(metadata_file)
        for row in reader:
            cleaned_row = {
                k.strip(): (v.strip() if isinstance(v, str) else "")
                for k, v in row.items()
                if k
            }
            table_name = cleaned_row.get("table_name", "")
            if table_name:
                descriptions[table_name] = cleaned_row.get("description", "")

    return descriptions


def setup_and_import():
    # 1. Connect and Create Tables
    conn = sqlitecloud.connect(conn_str)
    table_descriptions = load_table_descriptions()
    
    # Create the metadata structure
    conn.execute("DROP TABLE IF EXISTS column_metadata;")
    conn.execute("DROP TABLE IF EXISTS table_metadata;")

    conn.execute("""
    CREATE TABLE table_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_name TEXT NOT NULL UNIQUE,
        description TEXT
    );
    """)

    conn.execute("""
    CREATE TABLE column_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_id INTEGER NOT NULL,
        original_name TEXT NOT NULL,
        friendly_name TEXT,
        description TEXT,
        data_format TEXT,
        value_description TEXT,
        FOREIGN KEY (table_id) REFERENCES table_metadata(id)
    );
    """)

    # 2. Process Files
    csv_files = [
        f for f in os.listdir(CSV_DIR)
        if f.endswith('.csv') and f != os.path.basename(TABLE_METADATA_FILE)
    ]
    
    for filename in csv_files:
        table_name = filename.replace('.csv', '')
        
        # Insert table entry
        conn.execute(
            "INSERT INTO table_metadata (table_name, description) VALUES (?, ?)",
            (table_name, table_descriptions.get(table_name, f"Formula 1 {table_name} records"))
        )
        
        # Get the assigned ID
        cursor = conn.execute("SELECT id FROM table_metadata WHERE table_name = ?", (table_name,))
        table_id = cursor.fetchone()[0]

        # Read CSV and Insert Columns
        file_path = os.path.join(CSV_DIR, filename)
        with open_csv_with_fallback(file_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Clean up keys in case of trailing spaces in CSV headers
                row = {
                    k.strip(): (v.strip() if isinstance(v, str) else "")
                    for k, v in row.items()
                    if k
                }
                
                conn.execute("""
                    INSERT INTO column_metadata 
                    (table_id, original_name, friendly_name, description, data_format, value_description)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    table_id,
                    row.get('original_column_name', ''),
                    row.get('column_name', ''),
                    row.get('column_description', ''),
                    row.get('data_format', ''),
                    row.get('value_description', '')
                ))
    
    conn.commit()
    print(f"Successfully indexed {len(csv_files)} tables.")
    conn.close()

if __name__ == "__main__":
    # load CSV_DIR from argument
    if len(sys.argv) < 2:
        print("Usage: python main.py <csv_directory>")
        sys.exit(1)
    CSV_DIR = sys.argv[1] if len(sys.argv) > 1
    # CSV_DIR = "/home/laksh/Downloads/data_minidev/MINIDEV/dev_databases/formula_1/database_description"
    TABLE_METADATA_FILE = os.path.join(CSV_DIR, "table_metadata_file.csv")
    setup_and_import()
