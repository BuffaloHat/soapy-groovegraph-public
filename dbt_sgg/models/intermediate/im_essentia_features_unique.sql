{{ config(materialized='view') }}

-- Found duplicates audio files in Apple Music (3 total)
-- Cleanup duplicates
-- One row per file_hash (dedup across duplicate audio files)
-- Used for Qdrant analysis downstream

with ranked as (
    select
        s.*,
        row_number() over (
            partition by file_hash
            order by coalesce(duration_sec, 0) desc, json_path
        ) as rn
    from {{ ref('stg_essentia_features') }} s
    where file_hash is not null
)
select *
from ranked
where rn = 1

