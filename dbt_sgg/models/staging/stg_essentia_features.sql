{{ config(materialized='view') }}

-- Staging view over Essentia feature dataset
-- Casts important columns to stable types and adds small conveniences

with src as (
    select * from {{ ref('ext_essentia_features') }}
), typed as (
    select
        cast(artist as varchar)               as artist,
        cast(album as varchar)                as album,
        cast(title as varchar)                as title,
        cast(date as varchar)                 as date,
        cast(genre as varchar)                as genre,
        cast(genre_csv as varchar)            as genre_csv,
        cast(albumartist as varchar)          as albumartist,
        cast(composer as varchar)             as composer,
        try_cast(tracknumber as integer)      as tracknumber,
        try_cast(discnumber as integer)       as discnumber,
        cast(file_name as varchar)            as file_name,

        try_cast(duration_sec as double)      as duration_sec,
        try_cast(sample_rate as bigint)       as sample_rate,
        try_cast(bit_rate as bigint)          as bit_rate,
        try_cast(channels as bigint)          as channels,
        cast(codec as varchar)                as codec,
        try_cast(lossless as boolean)         as lossless,
        try_cast(replay_gain as double)       as replay_gain,

        try_cast(bpm as double)               as bpm,
        try_cast(danceability as double)      as danceability,
        try_cast(onset_rate as double)        as onset_rate,
        try_cast(beats_count as bigint)       as beats_count,
        try_cast(bpm_peak1 as bigint)         as bpm_peak1,
        try_cast(bpm_peak2 as bigint)         as bpm_peak2,

        cast(key_key as varchar)              as key_key,
        cast(key_scale as varchar)            as key_scale,
        try_cast(key_strength as double)      as key_strength,
        try_cast(tuning_frequency as double)  as tuning_frequency,
        try_cast(equal_tempered_deviation as double) as equal_tempered_deviation,
        try_cast(chords_changes_rate as double) as chords_changes_rate,

        try_cast(loudness_integrated as double) as loudness_integrated,
        try_cast(loudness_range as double)    as loudness_range,
        try_cast(avg_loudness as double)      as avg_loudness,
        try_cast(dyn_complexity as double)    as dyn_complexity,

        try_cast(spectral_rms as double)      as spectral_rms,
        try_cast(spectral_centroid as double) as spectral_centroid,
        try_cast(spectral_flux as double)     as spectral_flux,
        try_cast(spectral_spread as double)   as spectral_spread,
        try_cast(pitch_salience as double)    as pitch_salience,
        try_cast(hfc as double)               as hfc,
        try_cast(zcr as double)               as zcr,

        -- keep vectors as lists
        mfcc_mean,
        thpcp,

        -- placeholders
        cast(voice_instrumental_value as varchar) as voice_instrumental_value,
        try_cast(voice_instrumental_prob_vocal as double) as voice_instrumental_prob_vocal,
        try_cast(voice_instrumental_prob_instrumental as double) as voice_instrumental_prob_instrumental,
        try_cast(mood_happy as double)         as mood_happy,
        try_cast(mood_relaxed as double)       as mood_relaxed,
        try_cast(mood_party as double)         as mood_party,
        cast(mood_model as varchar)            as mood_model,
        cast(mood_version as varchar)          as mood_version,

        -- provenance
        cast(features_version as varchar)      as features_version,
        cast(essentia_version as varchar)      as essentia_version,
        cast(extracted_with as varchar)        as extracted_with,
        cast(file_hash as varchar)             as file_hash,
        cast(json_path as varchar)             as json_path,
        cast(extracted_at as varchar)          as extracted_at,

        -- helpers
        try_cast(duration_sec as double)/60.0  as duration_min
    from src
)
-- Exclude non-music genres that add noise to audio similarity and RAG results
select * from typed
where coalesce(genre, '') not in ('Comedy', 'Books & Spoken')

