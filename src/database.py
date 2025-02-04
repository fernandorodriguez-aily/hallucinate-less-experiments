import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def initialize_db(db_path: Union[str, Path]):
    """
    Initialize the SQLite database and create the table if it doesn't exist.

    Args:
        db_path (Union[str, Path]): Path to the SQLite database file as a string
                                    or Path object.
    """
    # Convert db_path to a Path object if it's a string
    db_path = Path(db_path) if isinstance(db_path, str) else db_path

    # Create the directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        print(f"Database already exists at {db_path}")
    else:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER,
                file TEXT,
                model TEXT,
                title TEXT,
                reasoning TEXT,
                structured_reasoning TEXT,
                relevance_label TEXT,
                relevance_label_gt TEXT,
                PRIMARY KEY (id, file, model)
            )
            """
        )
        conn.commit()
        conn.close()
        print(f"Database initialized successfully at {db_path}")


def insert_row(
    db_path: Union[str, Path], row_data: Dict[str, Any], debug: bool = False
) -> None:
    """
    Insert a new row into the 'articles' table.

    Args:
        db_path (Union[str, Path]): Path to the SQLite database file as a string
                                    or Path object.
        row_data (Dict[str, Any]): A dictionary containing the data to insert.
                                   Keys should match the column names in the table.
                                   Example: {
                                       "title": "Sample Title",
                                       "reasoning": "Sample Reasoning",
                                       "relevance_label": "Sample Label"
                                   }
    """
    # Convert db_path to a Path object if it's a string
    db_path = Path(db_path) if isinstance(db_path, str) else db_path

    # Ensure the database exists
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Initialize it first."
        )

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Prepare the SQL query
    columns = ", ".join(row_data.keys())
    placeholders = ", ".join("?" * len(row_data))
    query = f"INSERT INTO results ({columns}) VALUES ({placeholders})"

    # Execute the query with the row data
    cursor.execute(query, tuple(row_data.values()))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()
    if debug:
        print(f"Row inserted successfully into {db_path}")


def get_id_file_model_combinations(db_path: Union[str, Path]) -> List[tuple]:
    """
    Fetch all existing ID, file, model combinations from the database.

    Args:
        db_path (Union[str, Path]): Path to the SQLite database file as a string
                                    or Path object.

    Returns:
        List[tuple]: A list of tuples containing (id, file) combinations.
    """
    # Convert db_path to a Path object if it's a string
    db_path = Path(db_path) if isinstance(db_path, str) else db_path

    # Ensure the database exists
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Initialize it first."
        )

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the query to fetch id and file combinations
    cursor.execute("SELECT id, file, model FROM results")

    # Fetch all results
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    return results


def get_rows(
    db_path: Union[str, Path],
    file_name: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch all rows from the 'results' table with optional filtering.

    Args:
        db_path (Union[str, Path]): Path to the SQLite database file.
        file_name (Optional[str]): Optional filter for the 'file' column.
        model (Optional[str]): Optional filter for the 'model' column.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing rows.
    """
    # Convert db_path to a Path object if it's a string
    db_path = Path(db_path) if isinstance(db_path, str) else db_path

    # Ensure the database exists
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Initialize it first."
        )

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Build query with optional filters
    query = "SELECT * FROM results"
    params = []

    conditions = []
    if file_name:
        conditions.append("file = ?")
        params.append(file_name)
    if model:
        conditions.append("model = ?")
        params.append(model)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    # Execute the query
    cursor.execute(query, params)

    # Fetch all results
    rows = cursor.fetchall()

    # Get column names
    column_names = [desc[0] for desc in cursor.description]

    # Close the connection
    conn.close()

    # Convert rows to a list of dictionaries
    return [dict(zip(column_names, row)) for row in rows]
