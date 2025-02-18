#include "SpnWrapper.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <mutable/mutable.hpp>
#include <mutable/util/Diagnostic.hpp>

using namespace m;
using namespace Eigen;
using namespace std;

SpnWrapper SpnWrapper::learn_spn_csv(const std::string &csv_file,
                                    std::vector<Spn::LeafType> leaf_types,
                                    const std::vector<std::size_t>& primary_key_columns)
{
    // Read CSV file
    std::ifstream file(csv_file);

    if (!file.is_open()) {
        cerr << "Error opening file: " << csv_file << endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    // Read headers
    std::vector<std::string> headers;
    std::getline(file, line);
    std::stringstream header_stream(line);
    std::string header;
    while(std::getline(header_stream, header, ',')) {
        headers.push_back(header);
    }

    // Read data rows
    std::vector<std::vector<std::string>> rows;
    while(std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream line_stream(line);
        std::string cell;
        while(std::getline(line_stream, cell, ',')) {
            row.push_back(cell);
        }
        rows.push_back(row);
    }

    const std::size_t num_columns = headers.size();
    const std::size_t num_rows = rows.size();

    leaf_types.resize(num_columns - primary_key_columns.size(), Spn::AUTO);

    MatrixXf data(num_rows, num_columns - primary_key_columns.size());
    MatrixXi null_matrix = MatrixXi::Zero(data.rows(), data.cols());
    std::unordered_map<ThreadSafePooledString, unsigned> attribute_to_id;

    // Create attribute mapping (excluding primary keys)
    std::size_t data_col = 0;
    for(std::size_t csv_col = 0; csv_col < num_columns; csv_col++) {
        if(std::find(primary_key_columns.begin(), primary_key_columns.end(), csv_col) != primary_key_columns.end()) {
            continue;
        }

        attribute_to_id.emplace(headers[csv_col], data_col);
        data_col++;
    }

    // Process each row
    for(std::size_t row_idx = 0; row_idx < num_rows; row_idx++) {
        const auto& row = rows[row_idx];
        data_col = 0;

        for(std::size_t csv_col = 0; csv_col < num_columns; csv_col++) {
            if(std::find(primary_key_columns.begin(), primary_key_columns.end(), csv_col) != primary_key_columns.end()) {
                continue;
            }

            const std::string& value = row[csv_col];
            const std::size_t col = data_col++;

            // Handle null values (empty string)
            if(value.empty()) {
                null_matrix(row_idx, col) = 1;
                data(row_idx, col) = 0;
                continue;
            }

            // Determine data type handling
            if(leaf_types[col] == Spn::CONTINUOUS ||
              (leaf_types[col] == Spn::AUTO && value.find('.') != std::string::npos))
            {
                // Handle numeric types
                try {
                    data(row_idx, col) = std::stof(value);
                } catch(...) {
                    null_matrix(row_idx, col) = 1;
                    data(row_idx, col) = 0;
                }
            }
            else if(leaf_types[col] == Spn::DISCRETE || leaf_types[col] == Spn::AUTO) {
                // Handle discrete values
                try {
                    data(row_idx, col) = std::stoi(value);
                } catch(...) {
                    // Fallback to hashing if not numeric
                    data(row_idx, col) = float(std::hash<std::string>{}(value));
                }
            }
        }
    }

    return SpnWrapper(Spn::learn_spn(data, null_matrix, leaf_types), std::move(attribute_to_id));
}

unordered_map<string, SpnWrapper*> SpnWrapper::learn_spn_from_csvs(const vector<string> &csv_files, unordered_map<string, vector<Spn::LeafType>> leaf_types)
{
    unordered_map<string, SpnWrapper*> spns;

    for (const auto &csv_file : csv_files) {
        string table_name = csv_file.substr(csv_file.find_last_of("/") + 1);
        spns.emplace(table_name, new SpnWrapper(learn_spn_csv(csv_file, leaf_types[table_name])));
    }

    return spns;
}
