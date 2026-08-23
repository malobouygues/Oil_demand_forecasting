-- Regional demand from the raw JODI extract.
--
-- The source is one 650 MB csv holding every flow, unit and product JODI collects. It is
-- queried in place rather than loaded: keep the demand flow in thousand barrels per day,
-- map the seven products of the paper's Table 1, map reporting countries to its seven
-- regions, and sum.
--
-- Reads three relations that data.py registers on the connection:
--   jodi_raw   the csv, as text
--   products   JODI product code -> d1..d7
--   countries  ISO-2 -> region, for the countries the paper names
--
-- reporters comes back with the total on purpose. A regional figure is the sum of whoever
-- reported that month, so a country leaving JODI looks exactly like a fall in demand.
WITH jodi AS (
    SELECT
        REF_AREA                        AS country,
        CAST(TIME_PERIOD || '-01' AS DATE) AS month,
        ENERGY_PRODUCT                  AS code,
        TRY_CAST(OBS_VALUE AS DOUBLE)   AS kbd   -- JODI writes '-' and 'N/A' for missing
    FROM jodi_raw
    WHERE FLOW_BREAKDOWN = 'TOTDEMO'
      AND UNIT_MEASURE   = 'KBD'
)
SELECT
    COALESCE(countries.region, 'other_non_oecd') AS region,   -- anything unnamed is the residual
    products.product,
    jodi.month,
    SUM(jodi.kbd)   AS kbd,
    COUNT(*)        AS reporters
FROM jodi
JOIN products  ON products.code     = jodi.code
LEFT JOIN countries ON countries.country = jodi.country
WHERE jodi.kbd IS NOT NULL
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;
