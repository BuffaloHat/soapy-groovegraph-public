{{ config(materialized='view') }}

-- External view over Essentia Parquet dataset produced by scripts/es_flatten_features.py
-- Uses absolute path for stability in DuckDB.

select *
from read_parquet('/Users/mattkennedy/Projects/sgg/data/raw/essentia/essentia_features_v1/*.parquet')

