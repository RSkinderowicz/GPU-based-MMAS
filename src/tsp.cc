/**
  @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
*/

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#include "tsp.h"


int32_t euc2d_distance(std::pair<double, double> p1,
                      std::pair<double, double> p2) {

    auto dx = p2.first - p1.first;
    auto dy = p2.second - p1.second;
    return static_cast<int32_t>(sqrt(dx * dx + dy * dy) + 0.5);
}


int32_t ceil_distance(std::pair<double, double> p1,
                      std::pair<double, double> p2) {

    auto dx = p2.first - p1.first;
    auto dy = p2.second - p1.second;
    return ceil(sqrt(dx * dx + dy * dy));
}


/**
 * Adapted from ACOTSP v1.03 by Thomas Stuetzle
 */
int32_t geo_distance (std::pair<double, double> p1,
                      std::pair<double, double> p2) {
    double deg, min;
    double lati, latj, longi, longj;
    double q1, q2, q3;
    long int dd;
    auto x1 = p1.first;
    auto x2 = p2.first; 
	auto y1 = p1.second;
    auto y2 = p2.second;

    deg = static_cast<int32_t>(x1);  // Truncate
    min = x1 - deg;
    lati = M_PI * (deg + 5.0 * min / 3.0) / 180.0;
    deg = static_cast<int32_t>(x2);
    min = x2 - deg;
    latj = M_PI * (deg + 5.0 * min / 3.0) / 180.0;

    deg = static_cast<int32_t>(y1);
    min = y1 - deg;
    longi = M_PI * (deg + 5.0 * min / 3.0) / 180.0;
    deg = static_cast<int32_t>(y2);
    min = y2 - deg;
    longj = M_PI * (deg + 5.0 * min / 3.0) / 180.0;

    q1 = cos (longi - longj);
    q2 = cos (lati - latj);
    q3 = cos (lati + latj);
    dd = (int) (6378.388 * acos (0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
    return dd;
}


/**
 * Adapted from ACOTSP v1.03 by Thomas Stuetzle
 */
int32_t att_distance (std::pair<double, double> p1,
                      std::pair<double, double> p2) 
{
    double xd = p1.first - p2.first;
    double yd = p1.second - p2.second;
    double rij = sqrt ((xd * xd + yd * yd) / 10.0);
    double tij = static_cast<int32_t>(rij);
    long int dij;

    if (tij < rij)
        dij = (int) tij + 1;
    else
        dij = (int) tij;
    return dij;
}


/**
 * Tries to load a Traveling Salesman Problem (or ATSP) instance in TSPLIB
 * format from file at 'path'. Only the instances with 'EDGE_WEIGHT_TYPE:
 * EUC_2D' or 'EXPLICIT' are supported.
 *
 * Throws runtime_error if the file is in unsupported format or if an error was
 * encountered.
 *
 * Returns the loaded problem instance.
 */
ProblemInstance load_tsplib_instance(const char *path) {
    enum EdgeWeightFormat { UPPER_DIAG_ROW, LOWER_DIAG_ROW, UPPER_ROW, FUNCTION };

    using namespace std;

    ifstream in(path);

    if (!in.is_open()) {
        throw runtime_error(string("Cannot open TSP instance file: ") + path);
    }

    string line;

    uint32_t dimension = 0;
    vector<pair<double, double>> coords;
    vector<double> distances;
    EdgeWeightType edge_weight_type{ EUC_2D };
    bool is_symmetric = true;
    string name { path };
    EdgeWeightFormat edge_weight_format { UPPER_DIAG_ROW };

    while (getline(in, line)) {
        cout << "Read line: " << line << endl;
        if (line.find("TYPE") == 0) {
            if (line.find(" TSP") != string::npos) {
                is_symmetric = true;
            } else if (line.find(" ATSP") != string::npos) {
                is_symmetric = false;
            } else {
                throw runtime_error("Unknown problem type");
            }
        } else if (line.find("NAME") != string::npos) {
            istringstream line_in(line.substr(line.find(':') + 1));
            line_in >> name;
        } else if (line.find("DIMENSION") != string::npos) {
            istringstream line_in(line.substr(line.find(':') + 1));
            if (!(line_in >> dimension)) {
                throw runtime_error(string("Cannot read instance dimension"));
            }
        } else if (line.find("EDGE_WEIGHT_FORMAT") != string::npos) {
            if (line.find(" UPPER_DIAG_ROW") != string::npos) {
                edge_weight_format = UPPER_DIAG_ROW;
            } else if (line.find(" LOWER_DIAG_ROW") != string::npos) {
                edge_weight_format = LOWER_DIAG_ROW;
            } else if (line.find(" UPPER_ROW") != string::npos) {
                edge_weight_format = UPPER_ROW;
            } else if (line.find(" FUNCTION") != string::npos) {
                edge_weight_format = FUNCTION;
            } else {
                throw runtime_error(string("Unsupported edge weight format"));
            }
        } else if (line.find("EDGE_WEIGHT_TYPE") != string::npos) {
            if (line.find(" EUC_2D") != string::npos) {
                edge_weight_type = EUC_2D;
            } else if (line.find(" CEIL_2D") != string::npos) {
                edge_weight_type = CEIL_2D;
            } else if (line.find(" GEO") != string::npos) {
                edge_weight_type = GEO;
            } else if (line.find(" ATT") != string::npos) {
                edge_weight_type = ATT;
            } else if (line.find(" EXPLICIT") != string::npos) {
                edge_weight_type = EXPLICIT;
            } else {
                throw runtime_error(string("Unsupported edge weight type"));
            }
        } else if (line.find("NODE_COORD_SECTION") != string::npos) {
            coords.reserve(dimension * 2);
            while (getline(in, line)) {
                if (line.find("EOF") == string::npos) {
                    istringstream line_in(line);
                    uint32_t id;
                    pair<double, double> point;
                    line_in >> id >> point.first >> point.second;
                    if (line_in.bad()) {
                        cerr << "Error while reading coordinates";
                    }
                    coords.push_back(point);
                } else {
                    break;
                }
            }
        } else if (line.find("EDGE_WEIGHT_SECTION") != string::npos) {
            assert(dimension > 0);
            if (edge_weight_type != EXPLICIT) {
                throw runtime_error("Expected EXPLICIT edge weight type");
            }

            if (edge_weight_format == UPPER_DIAG_ROW) {
                distances.resize(dimension * dimension);

                uint32_t row = 0;
                uint32_t col = 0;
                while (row < dimension && getline(in, line)) {
                    istringstream line_in(line);
                    double distance;
                    while (line_in >> distance) {
                        distances.at(row * dimension + col) = distance;
                        distances.at(col * dimension + row) = distance;
                        ++col;
                        if (col == dimension) {
                            ++row;
                            col = row;
                        }
                    }
                }
            } else if (edge_weight_format == UPPER_ROW) {
                distances.resize(dimension * dimension);

                uint32_t row = 0;
                uint32_t col = 1;
                while (row < dimension && getline(in, line)) {
                    istringstream line_in(line);
                    double distance;
                    while (line_in >> distance) {
                        distances.at(row * dimension + col) = distance;
                        distances.at(col * dimension + row) = distance;
                        ++col;
                        if (col == dimension) {
                            ++row;
                            col = row + 1;
                        }
                    }
                }
            } else if (edge_weight_format == LOWER_DIAG_ROW) {
                distances.resize(dimension * dimension);

                uint32_t row = 0;
                uint32_t col = 0;
                while (row < dimension && getline(in, line)) {
                    istringstream line_in(line);
                    double distance;
                    while (line_in >> distance) {
                        distances.at(row * dimension + col) = distance;
                        distances.at(col * dimension + row) = distance;
                        ++col;
                        if (col > row) {
                            ++row;
                            col = 0;
                            if (row == dimension) {
                                break ;
                            }
                        }
                    }
                }
            } else {
                distances.reserve(dimension * dimension);
                while (getline(in, line)) {
                    if (line.find("EOF") != string::npos) {
                        break;
                    }
                    istringstream line_in(line);
                    double distance;
                    while (line_in >> distance) {
                        distances.push_back(distance);
                    }
                }
            }
            assert(distances.size() == dimension * dimension);
        }
    }
    in.close();

    assert(dimension > 2);

    std::cout  << "Finished loading instance" << std::endl;
    return ProblemInstance(dimension,
                           coords,
                           distances,
                           is_symmetric,
                           name,
                           edge_weight_type);
}


std::vector<std::vector<uint32_t>> 
compute_nn_lists(KDTree &kdtree, uint32_t nn_size = 32) {
    auto n = kdtree.get_points_count();

    std::vector<std::vector<uint32_t>> nn_lists(n);

    for (uint32_t i = 0; i < n; ++i) {
        auto &nn_list = nn_lists[i];
        nn_list.reserve(nn_size);

        for (uint32_t j = 0; j < nn_size; ++j) {
            auto pt_idx = kdtree.nn_bottom_up(i);
            nn_list.push_back(pt_idx);
            kdtree.delete_point(pt_idx);
        }
        for (auto pt_idx : nn_list) {
            kdtree.undelete_point(pt_idx);
        }
    }
    return nn_lists;
}


std::vector<uint32_t> build_nn_tour(KDTree &kdtree,
                                    uint32_t start_point_idx = 0) {
    auto n = kdtree.get_points_count();
    assert(n > 0);

    std::vector<uint32_t> tour(n);
    tour.clear();

    tour.push_back(start_point_idx);
    kdtree.delete_point(start_point_idx);

    for (uint32_t i = 1; i < n; ++i) {
        auto pt_idx = kdtree.nn(tour.back());
        tour.push_back(pt_idx);
        kdtree.delete_point(pt_idx);
    }

    for (auto it = tour.rbegin(); it != tour.rend(); ++it) {
        kdtree.undelete_point(*it);
    }
    return tour;
}
