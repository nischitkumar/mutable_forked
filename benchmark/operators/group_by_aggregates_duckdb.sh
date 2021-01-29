#!/bin/bash

# Define path to DuckDB CLI
DUCKDB=duckdb_cli

{ ${DUCKDB} | grep 'Run Time' | cut -d ' ' -f 4 | awk '{print $1 * 1000;}'; } << EOF
CREATE TABLE Distinct_i32 ( id INT, n1 INT, n10 INT, n100 INT, n1000 INT, n10000 INT, n100000 INT);
COPY Distinct_i32 FROM 'benchmark/operators/data/Distinct_i32.csv' ( HEADER );
.timer on
SELECT MIN(n100000) FROM Distinct_i32 GROUP BY n10;
SELECT MIN(n100000), MIN(n10000) FROM Distinct_i32 GROUP BY n10;
SELECT MIN(n100000), MIN(n10000), MIN(n1000) FROM Distinct_i32 GROUP BY n10;
SELECT MIN(n100000), MIN(n10000), MIN(n1000), MIN(n100) FROM Distinct_i32 GROUP BY n10;
EOF
