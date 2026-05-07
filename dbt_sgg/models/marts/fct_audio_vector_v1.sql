{{ config(materialized='view') }}

-- Fact view emitting ordered numeric vectors for audio similarity (v1)
-- Source: one row per file_hash from intermediate deduped features

with base as (
    select
        file_hash,
        artist, album, title, date, genre,
        key_key, key_scale, key_strength,
        features_version,
        -- core numeric features
        try_cast(danceability as double)      as danceability,
        try_cast(bpm as double)               as bpm,
        try_cast(loudness_integrated as double) as loudness_integrated,
        try_cast(loudness_range as double)    as loudness_range,
        try_cast(spectral_rms as double)      as spectral_rms,
        try_cast(spectral_centroid as double) as spectral_centroid,
        try_cast(spectral_flux as double)     as spectral_flux,
        try_cast(pitch_salience as double)    as pitch_salience,
        try_cast(hfc as double)               as hfc,
        try_cast(zcr as double)               as zcr,
        try_cast(onset_rate as double)        as onset_rate
    from {{ ref('im_essentia_features_unique') }}
), stats as (
    select
        avg(danceability)                       as danceability_mean,
        stddev_samp(danceability)               as danceability_std,
        avg(bpm)                                 as bpm_mean,
        stddev_samp(bpm)                         as bpm_std,
        avg(loudness_integrated)                as loudness_integrated_mean,
        stddev_samp(loudness_integrated)        as loudness_integrated_std,
        avg(loudness_range)                     as loudness_range_mean,
        stddev_samp(loudness_range)             as loudness_range_std,
        avg(spectral_rms)                       as spectral_rms_mean,
        stddev_samp(spectral_rms)               as spectral_rms_std,
        avg(spectral_centroid)                  as spectral_centroid_mean,
        stddev_samp(spectral_centroid)          as spectral_centroid_std,
        avg(spectral_flux)                      as spectral_flux_mean,
        stddev_samp(spectral_flux)              as spectral_flux_std,
        avg(pitch_salience)                     as pitch_salience_mean,
        stddev_samp(pitch_salience)             as pitch_salience_std,
        avg(hfc)                                as hfc_mean,
        stddev_samp(hfc)                        as hfc_std,
        avg(zcr)                                as zcr_mean,
        stddev_samp(zcr)                        as zcr_std,
        avg(onset_rate)                         as onset_rate_mean,
        stddev_samp(onset_rate)                 as onset_rate_std,
        avg(try_cast(key_strength as double))   as key_strength_mean,
        stddev_samp(try_cast(key_strength as double)) as key_strength_std
    from base
), vec as (
    select
        b.file_hash,
        b.artist, b.album, b.title, b.date, b.genre,
        b.key_key, b.key_scale, try_cast(b.key_strength as double) as key_strength,
        b.features_version,

        -- raw metrics
        b.danceability, b.bpm, b.loudness_integrated, b.loudness_range,
        b.spectral_rms, b.spectral_centroid, b.spectral_flux, b.pitch_salience,
        b.hfc, b.zcr, b.onset_rate,

        -- z-scores (dataset-wide; for testing/analysis)
        (b.danceability - s.danceability_mean) / nullif(s.danceability_std, 0)         as danceability_z,
        (b.bpm - s.bpm_mean) / nullif(s.bpm_std, 0)                                     as bpm_z,
        (b.loudness_integrated - s.loudness_integrated_mean) / nullif(s.loudness_integrated_std, 0) as loudness_integrated_z,
        (b.loudness_range - s.loudness_range_mean) / nullif(s.loudness_range_std, 0)    as loudness_range_z,
        (b.spectral_rms - s.spectral_rms_mean) / nullif(s.spectral_rms_std, 0)         as spectral_rms_z,
        (b.spectral_centroid - s.spectral_centroid_mean) / nullif(s.spectral_centroid_std, 0) as spectral_centroid_z,
        (b.spectral_flux - s.spectral_flux_mean) / nullif(s.spectral_flux_std, 0)       as spectral_flux_z,
        (b.pitch_salience - s.pitch_salience_mean) / nullif(s.pitch_salience_std, 0)    as pitch_salience_z,
        (b.hfc - s.hfc_mean) / nullif(s.hfc_std, 0)                                     as hfc_z,
        (b.zcr - s.zcr_mean) / nullif(s.zcr_std, 0)                                     as zcr_z,
        (b.onset_rate - s.onset_rate_mean) / nullif(s.onset_rate_std, 0)                as onset_rate_z,
        (try_cast(b.key_strength as double) - s.key_strength_mean) / nullif(s.key_strength_std, 0) as key_strength_z,

        -- ordered vectors for Qdrant (v1)
        list_value(
            b.danceability,
            b.bpm,
            b.loudness_integrated,
            b.loudness_range,
            b.spectral_rms,
            b.spectral_centroid,
            b.spectral_flux,
            b.pitch_salience,
            b.hfc,
            b.zcr,
            b.onset_rate,
            try_cast(b.key_strength as double)
        ) as vector_raw,
        list_value(
            (b.danceability - s.danceability_mean) / nullif(s.danceability_std, 0),
            (b.bpm - s.bpm_mean) / nullif(s.bpm_std, 0),
            (b.loudness_integrated - s.loudness_integrated_mean) / nullif(s.loudness_integrated_std, 0),
            (b.loudness_range - s.loudness_range_mean) / nullif(s.loudness_range_std, 0),
            (b.spectral_rms - s.spectral_rms_mean) / nullif(s.spectral_rms_std, 0),
            (b.spectral_centroid - s.spectral_centroid_mean) / nullif(s.spectral_centroid_std, 0),
            (b.spectral_flux - s.spectral_flux_mean) / nullif(s.spectral_flux_std, 0),
            (b.pitch_salience - s.pitch_salience_mean) / nullif(s.pitch_salience_std, 0),
            (b.hfc - s.hfc_mean) / nullif(s.hfc_std, 0),
            (b.zcr - s.zcr_mean) / nullif(s.zcr_std, 0),
            (b.onset_rate - s.onset_rate_mean) / nullif(s.onset_rate_std, 0),
            (try_cast(b.key_strength as double) - s.key_strength_mean) / nullif(s.key_strength_std, 0)
        ) as vector_z,
        'sgg_audio_v1' as vector_profile,
        12 as vector_dim
    from base b
    cross join stats s
)
select * from vec

