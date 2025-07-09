from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from corral.utils import create_vector_database

load_dotenv("../.env", override=True)


def main():
    # Files from the chembench commit 6d1a19ed62bc57f092ca5cad2c63ca57ee7551e7
    csv_files = list(
        Path("../materials_compatibility/processing_scripts").glob("*.csv")
    )

    for file in csv_files:
        material = file.split("/")[-1].split(".")[0].split("-")[0]
        df_mat_comp = pd.read_csv(file, header=1, sep=",")

        new_columns = {col: i + 1 for i, col in enumerate(df_mat_comp.columns)}
        df_mat_comp = df_mat_comp.rename(columns=new_columns)

        compatibility_entries = []
        for _, row in df_mat_comp.iterrows():
            chemical = row[1]
            condition = row[2]

            entry = f"{{Material {material} with chemical {chemical} have a compatibility: {condition}}}"
            compatibility_entries.append(entry)

        update_mode = "recreate" if file == csv_files[0] else "append"

        result = create_vector_database(
            chunks=compatibility_entries,
            collection_name="materials_compatibility",
            path="../vector_databases/materials_compatibility",
            chunk_size=8192,
            update_mode=update_mode,
        )

        logger.info(f"Processed {file} with {len(compatibility_entries)} entries.")
        logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
