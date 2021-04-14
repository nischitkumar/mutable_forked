#!/bin/bash

CSV="benchmark/tpc-h/data/lineitem.tbl"
NUM_ROWS=$(wc -l "${CSV}" | cut -f 1 -d ' ')
NUM_ROWS=$((NUM_ROWS-1))

# Define path to PostgreSQL CLI
POSTGRESQL=psql

trap 'exit' INT
{ ${POSTGRESQL} -U postgres ; } << EOF
DROP DATABASE IF EXISTS benchmark_tmp;
CREATE DATABASE benchmark_tmp;
\c benchmark_tmp
set jit=off;
CREATE TABLE Lineitem (
    l_orderkey      INT,
    l_partkey       INT,
    l_suppkey       INT,
    l_linenumber    INT,
    l_quantity      DECIMAL(10,2),
    l_extendedprice DECIMAL(10,2),
    l_discount      DECIMAL(10,2),
    l_tax           DECIMAL(10,2),
    l_returnflag    CHAR(1),
    l_linestatus    CHAR(1),
    l_shipdate      DATE,
    l_commitdate    DATE,
    l_receiptdate   DATE,
    l_shipinstruct  CHAR(25),
    l_shipmode      CHAR(10),
    l_comment       CHAR(44)
);
\copy Lineitem FROM '${CSV}' WITH DELIMITER '|' CSV HEADER;
\timing on
SELECT
        SUM(l_extendedprice * l_discount) AS revenue
FROM
        Lineitem
WHERE
        l_shipdate >= DATE '1994-01-01'
        AND l_shipdate < DATE '1995-01-01'
        AND l_discount > 0.05 AND l_discount < 0.07
        AND l_quantity < 24;
SELECT
        SUM(l_extendedprice * l_discount) AS revenue
FROM
        Lineitem
WHERE
        l_shipdate >= DATE '1994-01-01'
        AND l_shipdate < DATE '1995-01-01'
        AND l_discount > 0.05 AND l_discount < 0.07
        AND l_quantity < 24;
SELECT
        SUM(l_extendedprice * l_discount) AS revenue
FROM
        Lineitem
WHERE
        l_shipdate >= DATE '1994-01-01'
        AND l_shipdate < DATE '1995-01-01'
        AND l_discount > 0.05 AND l_discount < 0.07
        AND l_quantity < 24;
\timing off
EOF
