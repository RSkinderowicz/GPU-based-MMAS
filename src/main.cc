/**
  @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
*/
#include "docopt.h"
#include "mmas.h"
#include "utils.h"


static const char USAGE[] =
R"(GPU-based MMAS.

  Usage:
    mmas --instance=<path> [--alg=<str>] [--block-warps=<n>]
         [--trials=<n>] [--iter=<n>]
         [--ants=<n>] [--cand-list-size=<n>]
         [--rho=<n>] [--ls=<n>]
         [--seed=<n>] [--results-dir=<path>]
         [--ls-block-warps=<n>]
    mmas (-h | --help)
    mmas --version

  Options:
    --instance=<path>       Path to a TSP instance in TSPLIB format.
    --alg=<str>             Name of the alg. to run (
                            mmas_rwm_lc, mmas_rwm_ct, mmas_rwm_bt,
                            mmas_wrs_lc, mmas_wrs_ct, mmas_wrs_bt -
                            optionally with _cl suffix for candidate lists use
                            ) [default: mmas_rwm_bt_cl]
    --block-warps=<n>       Number of warps (each with 32 threads) per block
                            of the sol. build kernel [default: 1]
    --trials=<n>            Number of trials (alg. runs or repetitions)
                            [default: 1]
    --iter=<n>              Number of iterations per trial - if 0 is given then 
                            it is set so that 1 million sol. are build [default: 1000]
    --ants=<n>              Number of ants, if 0 then #ants equals the size
                            of a problem (TSP instance) [default: 100]
    --seed=<n>              Seed for the random number generator, if 0 time-based
                            value is used [default: 0]
    --cand-list-size=<n>    Size of the candidate lists used in the MMAS,
                            needs to be a multiple of 32 [default: 32]
    --results-dir=<path>    Where to save the file with results. If it does not
                            exist, it will be created [default: results]
    --ls=<n>                Local search alg.: 0 - none, 1 - 2-opt [default: 1]
    --ls-block-warps=<n>    # warps per block used by the LS, if 0 a heuristic is used [default: 0]
    --rho=<n>               How fast pheromone evaporates (1-rho) [default: 0.9]
    -h --help               Display help screen.
    --version               Display version.
)";


int main(int argc, char *argv[]) {
    using namespace std;

    auto args = docopt::docopt(
        USAGE,
        { argv + 1, argv + argc },
        true,  // show help if requested
        "GPU-based MMAS 1.0 by Rafal Skinderowicz");


    load_best_known_solutions("best-known.json");

    run_mmas_experiment(args);

    return EXIT_SUCCESS;
}