"""Extract GFlowNet results from an sqlite database."""

import sqlite3
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tap import tapify


def extract_results(results_path: Path, save_path: Path) -> None:
    """Extract GFlowNet results from an sqlite database and save to a CSV file.

    :param results_path: Path to the sqlite database
    :param save_path: Path to the CSV file
    """
    # Read the results from the sqlite database
    conn = sqlite3.connect(results_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results")
    results = cursor.fetchall()

    # Convert the results to a pandas DataFrame
    data = pd.DataFrame(results, columns=[col[0] for col in cursor.description])
    conn.close()

    # Process the data
    data = data.rename(
        columns={
            "smi": "smiles",
            "fr_0": "S. aureus",
            "fr_1": "Solubility",
            "fr_2": "sa_score",
            "fr_3": "molecular_weight",
        }
    )
    data["Solubility"] = 14 * data["Solubility"] - 10
    data["sa_score"] = -9 * data["sa_score"] + 10
    data["molecular_weight"] = [Descriptors.MolWt(Chem.MolFromSmiles(smiles)) for smiles in data["smiles"]]

    # Save the data to a CSV file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(save_path, index=False)


if __name__ == "__main__":
    tapify(extract_results)
