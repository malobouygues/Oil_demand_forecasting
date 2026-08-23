-- Demand is thousand barrels per day, the unit the paper reports. Every predictor is in
-- the unit of its own source: they are never added together, only fed to a tree.
CREATE TABLE demand (
    region    VARCHAR NOT NULL,
    product   VARCHAR NOT NULL,   -- d1_lpg .. d7_other, paper Table 1
    month     DATE    NOT NULL,
    kbd       DOUBLE  NOT NULL,
    reporters INTEGER NOT NULL,   -- countries in the sum that month
    PRIMARY KEY (region, product, month)
);

-- One row per published observation. period_end closes the period the number describes,
-- available_at is the day it could first be read; the gap between them is the whole
-- point-in-time argument.
CREATE TABLE predictors (
    region       VARCHAR NOT NULL,
    variable     VARCHAR NOT NULL,   -- paper Table 2 name
    source       VARCHAR NOT NULL,   -- fred, oecd or worldbank
    period_end   DATE    NOT NULL,
    available_at DATE    NOT NULL,
    value        DOUBLE  NOT NULL,
    PRIMARY KEY (region, variable, period_end)
);

-- The modelling table, panel, is not declared here. It is demand joined to whatever had
-- been published by each month end, one column per predictor, so its columns follow
-- prepare.predictors() rather than a fixed contract; data.py creates it from the frame.
-- The two tables above are the contract, and both are typed and keyed.
