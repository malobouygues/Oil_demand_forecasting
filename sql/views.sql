-- Annual regional demand in mb/d next to the number of reporting countries. Reading the
-- two together is the coverage control: other non-OECD loses reporters through the sample,
-- so its demand falls for a reason that has nothing to do with oil.
CREATE OR REPLACE VIEW v_regional_demand AS
SELECT
    YEAR(month)                       AS year,
    region,
    SUM(kbd) / 12 / 1000              AS mb_per_day,
    ROUND(AVG(reporters))             AS reporters
FROM demand
GROUP BY 1, 2
ORDER BY 1, 2;

-- Share of each region's barrels by product, five-year steps. Diesel-heavy Europe against
-- gasoline-heavy America shows up here, and so would a mapping error.
CREATE OR REPLACE VIEW v_product_mix AS
SELECT
    YEAR(month) AS year,
    region,
    product,
    100 * SUM(kbd) / SUM(SUM(kbd)) OVER (PARTITION BY YEAR(month), region) AS pct
FROM demand
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

-- How stale each predictor is when it reaches a panel row: the median gap in months
-- between the period a number describes and the day it was published.
CREATE OR REPLACE VIEW v_publication_lag AS
SELECT
    variable,
    source,
    COUNT(*)                                               AS observations,
    ROUND(MEDIAN(DATE_DIFF('month', period_end, available_at))) AS lag_months
FROM predictors
GROUP BY 1, 2
ORDER BY lag_months DESC, variable;
