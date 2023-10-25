#include <mutable/IR/HeuristicSearchPlanEnumerator.hpp>

#include <cstring>
#include <execution>
#include <functional>
#include <memory>
#include <mutable/catalog/Catalog.hpp>
#include <mutable/Options.hpp>
#include <mutable/util/fn.hpp>


using namespace m;
using namespace m::pe;
using namespace m::pe::hs;


/*======================================================================================================================
 * Options
 *====================================================================================================================*/

namespace {

///> command-line options for the HeuristicSearchPlanEnumerator
namespace options {

/** The heuristic search vertex to use. */
const char *vertex = "SubproblemsArray";
/** The vertex expansion to use. */
const char *expand = "BottomUpComplete";
/** The heuristic function to use. */
const char *heuristic = "zero";
/** The search method to use. */
const char *search = "AStar";
/** The weighting factor to use. */
float weighting_factor = 1.f;
/** Whether to compute an initial upper bound for cost-based pruning. */
bool initialize_upper_bound = false;

}

}


/*======================================================================================================================
 * Helper functions
 *====================================================================================================================*/

namespace {

template<typename State>
std::array<Subproblem, 2> delta(const State &before_join, const State &after_join)
{
    std::array<Subproblem, 2> delta;
    M_insist(before_join.size() == after_join.size() + 1);
    auto after_it = after_join.cbegin();
    auto before_it = before_join.cbegin();
    auto out_it = delta.begin();

    while (out_it != delta.end()) {
        M_insist(before_it != before_join.cend());
        if (after_it == after_join.cend()) {
            *out_it++ = *before_it++;
        } else if (*before_it == *after_it) {
            ++before_it;
            ++after_it;
        } else {
            *out_it++ = *before_it++;
        }
    }
    return delta;
}

template<typename PlanTable, typename State>
void reconstruct_plan_bottom_up(const State &state, PlanTable &PT, const QueryGraph &G,const CardinalityEstimator &CE,
                                const CostFunction &CF)
{
    static cnf::CNF condition; // TODO: use join condition

    const State *parent = state.parent();
    if (not parent) return;
    reconstruct_plan_bottom_up(*parent, PT, G, CE, CF); // solve recursively
    const auto D = delta(*parent, state); // find joined subproblems
    PT.update(G, CE, CF, D[0], D[1], condition); // update plan table
}

template<typename PlanTable, typename State>
void reconstruct_plan_top_down(const State &goal, PlanTable &PT, const QueryGraph &G, const CardinalityEstimator &CE,
                               const CostFunction &CF)
{
    const State *current = &goal;
    static cnf::CNF condition; // TODO: use join condition

    while (current->parent()) {
        const State *parent = current->parent();
        const auto D = delta(*current, *parent); // find joined subproblems
        PT.update(G, CE, CF, D[0], D[1], condition); // update plan table
        current = parent; // advance
    }
}

}


/*======================================================================================================================
 * Heuristic Search States: class/static fields
 *====================================================================================================================*/

namespace m::pe::hs {
namespace search_states {

SubproblemsArray::allocator_type SubproblemsArray::allocator_;
SubproblemTableBottomUp::allocator_type SubproblemTableBottomUp::allocator_;
EdgesBottomUp::allocator_type EdgesBottomUp::allocator_;
EdgePtrBottomUp::Scratchpad EdgePtrBottomUp::scratchpad_;

}
}

namespace m::pe::hs {

template<
    typename PlanTable,
    typename State,
    typename Expand,
    template<typename, typename, typename> typename Heuristic,
    template<typename, typename, typename, typename...> typename Search
>
bool heuristic_search(PlanTable &PT, const QueryGraph &G, const AdjacencyMatrix &M, const CostFunction &CF,
                      const CardinalityEstimator &CE, ai::SearchConfiguration config)
{
    using H = Heuristic<PlanTable, State, Expand>;
    State::RESET_STATE_COUNTERS();

    using search_algorithm = Search<
        State, Expand, H,
        /*----- context -----*/
        PlanTable&,
        const QueryGraph&,
        const AdjacencyMatrix&,
        const CostFunction&,
        const CardinalityEstimator&
    >;

    search_algorithm S(PT, G, M, CF, CE);

    if (Options::Get().statistics)
        std::cout << "initial upper bound is " << config.upper_bound << std::endl;

    try {
        State initial_state = Expand::template Start<State>(PT, G, M, CF, CE);
        H h(PT, G, M, CF, CE);
        const State &goal = S.search(
            /* initial_state= */ std::move(initial_state),
            /* expand=        */ Expand{},
            /* heuristic=     */ h,
            /* config=        */ config,
            /*----- Context -----*/
            PT, G, M, CF, CE
        );
        if (Options::Get().statistics)
            S.dump(std::cout);

        /*----- Reconstruct the plan from the found path to goal. -----*/
        if constexpr (std::is_base_of_v<expansions::TopDown, Expand>) {
            reconstruct_plan_top_down(goal, PT, G, CE, CF);
        } else {
            static_assert(std::is_base_of_v<expansions::BottomUp, Expand>, "unexpected expansion");
            reconstruct_plan_bottom_up(goal, PT, G, CE, CF);
        }
    } catch (std::logic_error err) {
        if constexpr (not search_algorithm::use_beam_search) {
            if (not Options::Get().quiet)
                std::cout << "search did not reach a goal state, fall back to DPccp" << std::endl;
            DPccp{}(G, CF, PT);
        }
    }
#ifdef COUNTERS
    if (Options::Get().statistics) {
        std::cout <<   "Vertices generated: " << State::NUM_STATES_GENERATED()
                  << "\nVertices expanded: " << State::NUM_STATES_EXPANDED()
                  << "\nVertices constructed: " << State::NUM_STATES_CONSTRUCTED()
                  << "\nVertices disposed: " << State::NUM_STATES_DISPOSED()
                  << std::endl;
    }
#endif
    return true;
}

}

namespace {

template<
    typename PlanTable,
    typename State,
    typename Expand,
    template<typename, typename, typename> typename Heuristic,
    template<typename, typename, typename, typename...> typename Search
>
bool heuristic_search_helper(const char *vertex_str, const char *expand_str, const char *heuristic_str,
                             const char *search_str, PlanTable &PT, const QueryGraph &G, const AdjacencyMatrix &M,
                             const CostFunction &CF, const CardinalityEstimator &CE)
{
    if (streq(options::vertex,    vertex_str   ) and
        streq(options::expand,    expand_str   ) and
        streq(options::heuristic, heuristic_str) and
        streq(options::search,    search_str   ))
    {
        ai::SearchConfiguration config;

        if (options::initialize_upper_bound) {
            config.upper_bound = [&]() {
                /*----- Run GOO to compute upper bound of plan cost. -----*/
                /* Beam search becomes incomplete if an upper bound derived from an actual plan is provided.  Beam
                 * search may never find that plan or any better plan, and hence return without finding a path. In this
                 * case, the initial plan found by GOO is used. */
                GOO Goo;
                Goo(G, CF, PT);
                return PT.get_final().cost;
            }();
        }

        config.weighting_factor = options::weighting_factor;

        return m::pe::hs::heuristic_search<
            PlanTable,
            State,
            Expand,
            Heuristic,
            Search
        >(PT, G, M, CF, CE, config);
    }
    return false;

}

}

namespace m::pe::hs {

template<typename PlanTable>
void HeuristicSearch::operator()(enumerate_tag, PlanTable &PT, const QueryGraph &G, const CostFunction &CF) const
{
    Catalog &C = Catalog::Get();
    auto &CE = C.get_database_in_use().cardinality_estimator();
    const AdjacencyMatrix &M = G.adjacency_matrix();


#define HEURISTIC_SEARCH(STATE, EXPAND, HEURISTIC, CONFIG) \
    if (heuristic_search_helper<PlanTable, \
                                search_states::STATE, \
                                expansions::EXPAND, \
                                heuristics::HEURISTIC, \
                                config::CONFIG \
                               >(#STATE, #EXPAND, #HEURISTIC, #CONFIG, PT, G, M, CF, CE)) \
    { \
        goto matched_heuristic_search; \
    }

    // bottom-up
    //   zero
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   zero,                           AStar                           )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   zero,                           AStar_with_cbp                  )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   zero,                           beam_search            )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   zero,                           beam_search_with_cbp   )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   zero,                           dynamic_beam_search    )

    //   sum
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   sum,                            AStar                           )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   sum,                            lazyAStar                       )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   sum,                            beam_search            )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   sum,                            dynamic_beam_search    )

    //   scaled_sum
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   scaled_sum,                     AStar                           )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   scaled_sum,                     beam_search            )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   scaled_sum,                     dynamic_beam_search    )

    //   avg_sel
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   avg_sel,                        AStar                           )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   avg_sel,                        beam_search            )

    //   product
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   product,                        AStar                           )

    //   GOO
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   GOO,                            AStar                           )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   GOO,                            beam_search            )
    HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   GOO,                            dynamic_beam_search    )

    // HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   bottomup_lookahead_cheapest,    AStar                           )
    // HEURISTIC_SEARCH(   SubproblemsArray,   BottomUpComplete,   perfect_oracle,                 AStar                           )

    // top-down
    //   zero
    HEURISTIC_SEARCH(   SubproblemsArray,   TopDownComplete,    zero,                           AStar                           )
    HEURISTIC_SEARCH(   SubproblemsArray,   TopDownComplete,    zero,                           AStar_with_cbp                  )

    //   sqrt_sum
    HEURISTIC_SEARCH(   SubproblemsArray,   TopDownComplete,    sqrt_sum,                       AStar                           )

    //   sum
    HEURISTIC_SEARCH(   SubproblemsArray,   TopDownComplete,    sum,                            AStar                           )
    HEURISTIC_SEARCH(   SubproblemsArray,   TopDownComplete,    sum,                            AStar_with_cbp                  )

    //    GOO
    HEURISTIC_SEARCH(   SubproblemsArray,   TopDownComplete,    GOO,                            AStar                           )
    HEURISTIC_SEARCH(   SubproblemsArray,   TopDownComplete,    zero,                           beam_search            )
    HEURISTIC_SEARCH(   SubproblemsArray,   TopDownComplete,    sum,                            beam_search            )
    HEURISTIC_SEARCH(   SubproblemsArray,   TopDownComplete,    zero,                           dynamic_beam_search    )
    HEURISTIC_SEARCH(   SubproblemsArray,   TopDownComplete,    GOO,                            beam_search            )

    throw std::invalid_argument("illegal search configuration");
#undef HEURISTIC_SEARCH

matched_heuristic_search:;
#ifndef NDEBUG
    if (Options::Get().statistics) {
        auto plan_cost = [&PT]() -> double {
            const Subproblem left  = PT.get_final().left;
            const Subproblem right = PT.get_final().right;
            return PT[left].cost + PT[right].cost;
        };

        const double hs_cost = plan_cost();
        DPccp dpccp;
        dpccp(G, CF, PT);
        const double dp_cost = plan_cost();

        std::cout << "AI: " << hs_cost << ", DP: " << dp_cost << ", Δ " << hs_cost / dp_cost << 'x' << std::endl;
        if (hs_cost > dp_cost)
            std::cout << "WARNING: Suboptimal solution!" << std::endl;
    }
#endif
}

// explicit template instantiation
template void HeuristicSearch::operator()(enumerate_tag, PlanTableSmallOrDense &PT, const QueryGraph &G, const CostFunction &CF) const;
template void HeuristicSearch::operator()(enumerate_tag, PlanTableLargeAndSparse &PT, const QueryGraph &G, const CostFunction &CF) const;

}


namespace {

__attribute__((constructor(202)))
void register_heuristic_search_plan_enumerator()
{
    Catalog &C = Catalog::Get();
    C.register_plan_enumerator(
        "HeuristicSearch",
        std::make_unique<HeuristicSearch>(),
        "uses heuristic search to find a plan; "
        "found plans are optimal when the search method is optimal and the heuristic is admissible"
    );

    /*----- Command-line arguments -----------------------------------------------------------------------------------*/
    C.arg_parser().add<const char*>(
        /* group=       */ "HeuristicSearch",
        /* short=       */ nullptr,
        /* long=        */ "--hs-vertex",
        /* description= */ "the heuristic search vertex to use",
        [] (const char *str) { options::vertex = str; }
    );
    C.arg_parser().add<const char*>(
        /* group=       */ "HeuristicSearch",
        /* short=       */ nullptr,
        /* long=        */ "--hs-expand",
        /* description= */ "the vertex expansion to use",
        [] (const char *str) { options::expand = str; }
    );
    C.arg_parser().add<const char*>(
        /* group=       */ "HeuristicSearch",
        /* short=       */ nullptr,
        /* long=        */ "--hs-heuristic",
        /* description= */ "the heuristic function to use",
        [] (const char *str) { options::heuristic = str; }
    );
    C.arg_parser().add<const char*>(
        /* group=       */ "HeuristicSearch",
        /* short=       */ nullptr,
        /* long=        */ "--hs-search",
        /* description= */ "the search method to use",
        [] (const char *str) { options::search = str; }
    );
    C.arg_parser().add<float>(
        /* group=       */ "HeuristicSearch",
        /* short=       */ nullptr,
        /* long=        */ "--hs-wf",
        /* description= */ "the weighting factor for the heuristic value (defaults to 1)",
        [] (float wf) { options::weighting_factor = wf; }
    );
    C.arg_parser().add<bool>(
        /* group=       */ "HeuristicSearch",
        /* short=       */ nullptr,
        /* long=        */ "--hs-init-upper-bound",
        /* description= */ "greedily compute an initial upper bound for cost-based pruning",
        [] (bool) { options::initialize_upper_bound = true; }
    );
}

}
