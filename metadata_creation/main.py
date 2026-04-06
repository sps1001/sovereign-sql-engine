import csv
import io
import os
import reprlib
import sys

import sqlitecloud
from pinecone import Pinecone
from pinecone.exceptions import PineconeException
from dotenv import load_dotenv

load_dotenv()

# Configuration
metadata_conn_str = (
    f"sqlitecloud://{os.getenv('SQLITE_HOST')}:{os.getenv('SQLITE_PORT')}/"
    f"{os.getenv('SQLITE_METADATA_DB')}?apikey={os.getenv('SQLITE_API_KEY')}"
)
source_conn_str = (
    f"sqlitecloud://{os.getenv('SQLITE_HOST')}:{os.getenv('SQLITE_PORT')}/"
    f"{os.getenv('SQLITE_DB')}?apikey={os.getenv('SQLITE_API_KEY')}"
)
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "metadata")
PINECONE_EMBED_MODEL = os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2")


def open_csv_with_fallback(file_path):
    with open(file_path, mode="rb") as source_file:
        raw_bytes = source_file.read()

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return io.StringIO(raw_bytes.decode(encoding, errors="strict"))
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Unable to decode CSV file with supported encodings: {file_path}")


def clean_row(row):
    cleaned = {
        key.strip(): (value.strip() if isinstance(value, str) else "")
        for key, value in row.items()
        if key
    }
    overflow_values = row.get(None, [])
    if overflow_values:
        overflow_text = ", ".join(
            value.strip() for value in overflow_values if isinstance(value, str) and value.strip()
        )
        if overflow_text:
            if cleaned.get("description"):
                cleaned["description"] = f"{cleaned['description']}, {overflow_text}"
            elif cleaned.get("column_description"):
                cleaned["column_description"] = (
                    f"{cleaned['column_description']}, {overflow_text}"
                )
            else:
                cleaned["description"] = overflow_text

    return cleaned


def chunked(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def quote_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def load_table_descriptions():
    descriptions = {}

    with open_csv_with_fallback(TABLE_METADATA_FILE) as metadata_file:
        reader = csv.DictReader(metadata_file)
        for row in reader:
            cleaned_row = clean_row(row)
            table_name = cleaned_row.get("table_name", "")
            if table_name:
                descriptions[table_name] = cleaned_row.get("description", "")

    return descriptions


def fetch_column_example(source_conn, table_name, column_name):
    if not table_name or not column_name:
        return ""

    table_identifier = quote_identifier(table_name)
    column_identifier = quote_identifier(column_name)
    query = f"""
        SELECT {column_identifier}
        FROM {table_identifier}
        WHERE {column_identifier} IS NOT NULL
          AND CAST({column_identifier} AS TEXT) != ''
        GROUP BY {column_identifier}
        LIMIT 3
    """

    try:
        cursor = source_conn.execute(query)
        rows = cursor.fetchall()
    except Exception:
        return ""

    if not rows:
        return ""

    examples = []
    for row in rows:
        if not row:
            continue
        value = row[0]
        if value is None:
            continue
        examples.append(value)

    if not examples:
        return ""

    formatted_examples = [reprlib.repr(value) for value in examples]
    return "[" + ", ".join(formatted_examples) + "]"


def build_table_record(table_name, description):
    return {
        "id": f"table::{table_name}",
        "text": description.strip(),
        "metadata": {
            "category": "table",
            "name": table_name,
        },
    }


def build_column_record(table_name, row):
    column_name = row.get("original_column_name", "") or row.get("column_name", "")
    description_parts = []

    if row.get("column_description"):
        description_parts.append(row["column_description"])
    if row.get("value_description"):
        description_parts.append(row["value_description"])

    return {
        "id": f"column::{table_name}::{column_name}",
        "text": "\n".join(description_parts).strip(),
        "metadata": {
            "category": "col",
            "name": column_name,
            "table_name": table_name,
        },
    }


def get_pinecone_index():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_host = os.getenv("PINECONE_INDEX_HOST", "").strip()
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is required to index metadata in Pinecone.")

    if not pinecone_index_host and not pinecone_index_name:
        raise ValueError(
            "Set either PINECONE_INDEX_HOST or PINECONE_INDEX_NAME to target a Pinecone index."
        )

    pinecone_client = Pinecone(api_key=pinecone_api_key)

    if pinecone_index_host:
        return pinecone_client, pinecone_client.Index(host=pinecone_index_host)

    if not pinecone_client.has_index(pinecone_index_name):
        raise ValueError(f"Pinecone index '{pinecone_index_name}' does not exist.")

    return pinecone_client, pinecone_client.Index(name=pinecone_index_name)


def index_metadata_in_pinecone(records):
    pinecone_records = [record for record in records if record["text"].strip()]
    if not pinecone_records:
        print("No metadata descriptions found to index in Pinecone.")
        return

    pinecone_client, pinecone_index = get_pinecone_index()

    for record_batch in chunked(pinecone_records, 96):
        try:
            embeddings = pinecone_client.inference.embed(
                model=PINECONE_EMBED_MODEL,
                inputs=[record["text"] for record in record_batch],
                parameters={"input_type": "passage", "truncate": "END"},
            )
        except PineconeException as exc:
            raise ValueError(
                "Failed to generate Pinecone embeddings. "
                f"Check PINECONE_EMBED_MODEL='{PINECONE_EMBED_MODEL}'. "
                "This SDK supports Pinecone-hosted embedding models such as "
                "'llama-text-embed-v2'."
            ) from exc

        vectors = []
        for record, embedding in zip(record_batch, embeddings):
            vectors.append(
                {
                    "id": record["id"],
                    "values": embedding["values"],
                    "metadata": record["metadata"],
                }
            )

        pinecone_index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)

    print(
        f"Successfully indexed {len(pinecone_records)} metadata descriptions in Pinecone."
    )


def setup_and_import():
    metadata_conn = sqlitecloud.connect(metadata_conn_str)
    source_conn = sqlitecloud.connect(source_conn_str)
    table_descriptions = load_table_descriptions()
    pinecone_records = []

    metadata_conn.execute("DROP TABLE IF EXISTS column_metadata;")
    metadata_conn.execute("DROP TABLE IF EXISTS table_metadata;")

    metadata_conn.execute(
        """
        CREATE TABLE table_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL UNIQUE,
            description TEXT
        );
        """
    )

    metadata_conn.execute(
        """
        CREATE TABLE column_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_id INTEGER NOT NULL,
            original_name TEXT NOT NULL,
            friendly_name TEXT,
            description TEXT,
            data_format TEXT,
            value_description TEXT,
            example TEXT,
            FOREIGN KEY (table_id) REFERENCES table_metadata(id)
        );
        """
    )

    csv_files = sorted(
        [
            filename
            for filename in os.listdir(CSV_DIR)
            if filename.endswith(".csv") and filename != os.path.basename(TABLE_METADATA_FILE)
        ]
    )

    for filename in csv_files:
        table_name = filename.replace(".csv", "")
        table_description = table_descriptions.get(
            table_name, f"Formula 1 {table_name} records"
        )

        metadata_conn.execute(
            "INSERT INTO table_metadata (table_name, description) VALUES (?, ?)",
            (table_name, table_description),
        )
        pinecone_records.append(build_table_record(table_name, table_description))

        cursor = metadata_conn.execute(
            "SELECT id FROM table_metadata WHERE table_name = ?", (table_name,)
        )
        table_id = cursor.fetchone()[0]

        file_path = os.path.join(CSV_DIR, filename)
        with open_csv_with_fallback(file_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                cleaned_row = clean_row(row)
                example_value = fetch_column_example(
                    source_conn,
                    table_name,
                    cleaned_row.get("original_column_name", ""),
                )
                metadata_conn.execute(
                    """
                    INSERT INTO column_metadata
                    (
                        table_id,
                        original_name,
                        friendly_name,
                        description,
                        data_format,
                        value_description,
                        example
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        table_id,
                        cleaned_row.get("original_column_name", ""),
                        cleaned_row.get("column_name", ""),
                        cleaned_row.get("column_description", ""),
                        cleaned_row.get("data_format", ""),
                        cleaned_row.get("value_description", ""),
                        example_value,
                    ),
                )

                pinecone_records.append(build_column_record(table_name, cleaned_row))

    metadata_conn.commit()
    metadata_conn.close()
    source_conn.close()

    print(f"Successfully indexed {len(csv_files)} tables in SQLite Cloud.")
    index_metadata_in_pinecone(pinecone_records)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <csv_directory>")
        sys.exit(1)

    CSV_DIR = sys.argv[1]
    TABLE_METADATA_FILE = os.path.join(CSV_DIR, "table_metadata_file.csv")
    setup_and_import()
