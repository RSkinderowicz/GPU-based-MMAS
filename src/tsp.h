#pragma once


#include <vector>
#include <string>
#include <cassert>
#include <algorithm>

#include "kd_tree.h"

std::vector<std::vector<uint32_t>> 
compute_nn_lists(KDTree &kdtree, uint32_t nn_size);


std::vector<uint32_t> build_nn_tour(KDTree &kdtree, uint32_t start_point_idx);


enum EdgeWeightType { EUC_2D, EXPLICIT, GEO, ATT, CEIL_2D };


int32_t euc2d_distance(std::pair<double, double> p1,
                       std::pair<double, double> p2);


int32_t ceil_distance(std::pair<double, double> p1,
                      std::pair<double, double> p2);


/**
 * Adapted from ACOTSP v1.03 by Thomas Stuetzle
 */
int32_t geo_distance (std::pair<double, double> p1,
                      std::pair<double, double> p2);

/**
 * Adapted from ACOTSP v1.03 by Thomas Stuetzle
 */
int32_t att_distance (std::pair<double, double> p1,
                      std::pair<double, double> p2);


struct ProblemInstance {
    using dist_fn_t = int32_t (*)(std::pair<double, double> p1,
                                  std::pair<double, double> p2);

    uint32_t dimension_;
    bool is_symmetric_ = true;
    std::vector<std::pair<double, double>> coordinates_;
    std::vector<double> distance_matrix_;
    std::vector<std::vector<uint32_t>> nearest_neighbor_lists_;
    std::string name_;
    KDTree kd_tree_;
    EdgeWeightType edge_weight_type_ { EUC_2D };
    dist_fn_t calc_dist_ { nullptr };

    ProblemInstance(uint32_t dimension,
                    const std::vector<std::pair<double, double>> &coordinates,
                    const std::vector<double> &distance_matrix,
                    bool is_symmetric,
                    const std::string &name,
                    EdgeWeightType edge_weight_type)
        : dimension_(dimension),
          is_symmetric_(is_symmetric),
          coordinates_(coordinates),
          distance_matrix_(distance_matrix),
          name_(name),
          kd_tree_(coordinates),
          edge_weight_type_(edge_weight_type)
    {
        assert(dimension >= 2);

        switch(edge_weight_type_) {
            case EUC_2D  : calc_dist_ = euc2d_distance; break;
            case CEIL_2D : calc_dist_ = ceil_distance; break;
            case GEO     : calc_dist_ = geo_distance; break;
            case ATT     : calc_dist_ = att_distance; break;
            case EXPLICIT: calc_dist_ = nullptr; break;
            default: throw std::runtime_error(
                std::string("Unsupported edge weight type"));
        }
    }

    void init_nn_lists(uint32_t cand_list_size) {
        if ( (edge_weight_type_ == EUC_2D || edge_weight_type_ == CEIL_2D)
           && kd_tree_.get_points_count() > 0 ) {
            // Using kd-tree is much faster than the alternative
            nearest_neighbor_lists_ = compute_nn_lists(kd_tree_, cand_list_size);
        } else {
            init_nn_lists_naive(cand_list_size);
        }
    }

    void init_nn_lists_naive(uint32_t cand_list_size) {
        using namespace std;

        assert(dimension_ > 1);

        cand_list_size = min(dimension_ - 1, cand_list_size);

        nearest_neighbor_lists_.resize(dimension_);
        vector<uint32_t> neighbors(dimension_);
        for (uint32_t i = 0; i < dimension_; ++i) {
            neighbors[i] = i;
        }

        for (uint32_t node = 0; node < dimension_; ++node) {
            // This puts the closest cand_list_size + 1 nodes in front of
            // the array (and sorted)
            partial_sort(neighbors.begin(),
                         neighbors.begin() + std::min(cand_list_size + 2u, dimension_),
                         neighbors.end(),
                         [this, node](uint32_t a, uint32_t b) {
                             return this->get_distance(node, a) < this->get_distance(node, b);
                         });

            assert(get_distance(node, neighbors.at(0)) <=
                   get_distance(node, neighbors.at(1)));

            auto &nn_list = nearest_neighbor_lists_.at(node);
            nn_list.clear();
            nn_list.reserve(cand_list_size);
            uint32_t count = 0;
            for (uint32_t i = 0; count < cand_list_size; ++i) {
                if (neighbors[i] != node) { // node is not its own neighbor
                    nn_list.push_back(neighbors[i]);
                    ++count;
                }
            }
        }
    }

    double get_distance(uint32_t from, uint32_t to) const {
        assert((from < dimension_) && (to < dimension_));

        return ! distance_matrix_.empty() ? distance_matrix_[from * dimension_ + to]
                                          : calculate_distance(from, to);
    }

    /**
    This calculates distance between the given nodes (cities) of the TSP
    instance using the rounding method for the EUC_2D distance type from the
    TSPLIB repository.
    */
    double calculate_distance(uint32_t from, uint32_t to) const {
        assert(from < dimension_ && to < dimension_);
        assert(edge_weight_type_ == EXPLICIT 
               || coordinates_.size() == dimension_);
        assert(calc_dist_ != nullptr);

        return calc_dist_(coordinates_[from], coordinates_[to]);
    }


    void init_distance_matrix() {
        if (edge_weight_type_ == EXPLICIT) {
            return ;  // Already initialized, i.e. loaded from an instance file
        }
        distance_matrix_.resize(dimension_ * dimension_, 0);

        for (uint32_t i = 0; i < dimension_; ++i) {
            for (uint32_t j = (is_symmetric_ ? i + 1 : 0); j < dimension_; ++j) {
                if (i != j) {
                    auto distance = calculate_distance(i, j);
                    distance_matrix_[i * dimension_ + j] = distance;

                    if (is_symmetric_) {
                        distance_matrix_[j * dimension_ + i] = distance;
                    }
                }
            }
        }
    }


    const std::vector<uint32_t> &get_nearest_neighbors(uint32_t node) const {
        assert(node < nearest_neighbor_lists_.size());

        return nearest_neighbor_lists_[node];
    }

    double calculate_route_length(const std::vector<uint32_t> &route) const {
        double distance = 0;
        if (!route.empty()) {
            auto prev_node = route.back();
            for (auto node : route) {
                distance += get_distance(prev_node, node);
                prev_node = node;
            }
        }
        return distance;
    }

    bool is_route_valid(const std::vector<uint32_t> &route) const {
        if (route.size() != dimension_) {
            return false;
        }
        std::vector<uint8_t> was_visited(dimension_, false);
        for (auto node : route) {
            if (was_visited[node]) {
                return false;
            }
            was_visited[node] = true;
        }
        return true;
    }

    bool is_small() const { return dimension_ < (1024 * 2); }


    /** 
     * This creates a solution using nearest neighbor heuristic that always
     * selects a clostest of the (yet) unvisited nodes (cities).
     */
    std::vector<uint32_t> create_solution_nn(uint32_t start_node = 0) {
        if (kd_tree_.get_points_count() > 0) {
            return build_nn_tour(kd_tree_, start_node);
        }
        assert( nearest_neighbor_lists_.size() == dimension_ );

        std::vector<uint32_t> tour(dimension_);
        tour.clear();
        std::vector<int8_t> visited(dimension_, false);
        tour.push_back(start_node);
        visited[start_node] = true;
        auto curr_node = start_node;
        auto next_node = curr_node;
        for (uint32_t i = 0; i+1 < dimension_; ++i) {
            for (auto node : nearest_neighbor_lists_.at(curr_node)) {
                if (!visited.at(node)) {
                    next_node = node;
                    break ;
                }
            }
            if (next_node == curr_node) {  // Not found -- check all unvisited nodes
                double closest_dist = std::numeric_limits<double>::max();
                for (uint32_t node = 0; node < dimension_; ++node) {
                    if ( !visited.at(node) 
                      && (next_node == curr_node 
                          || get_distance(curr_node, node) < closest_dist) ) {

                        closest_dist = get_distance(curr_node, node);
                        next_node = node;
                    }
                }
            }
            assert(next_node != curr_node);
            curr_node = next_node;
            tour.push_back(curr_node);
            visited[curr_node] = true;
        }
        return tour;
    }
};


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
ProblemInstance load_tsplib_instance(const char *path);