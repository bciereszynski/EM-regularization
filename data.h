#ifndef GENERATE_DATA_H
#define GENERATE_DATA_H

#include <vector>
#include <string>

std::vector<std::vector<double> > generate_data(int n, int d, unsigned int seed = 0);

std::vector<std::vector<double> > load_data_from_file(const std::string &path, int &k);

#endif // GENERATE_DATA_H
