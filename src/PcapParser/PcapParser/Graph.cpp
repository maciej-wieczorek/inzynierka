#include "Defines.h"

#include "Graph.h"

std::ostream& operator<<(std::ostream& os, const Graph& graph)
{
    size_t i = 0;
    for (const auto& edge : graph.edgeList)
    {
        os << edge.indexStart << ',' << edge.indexEnd << ',' << edge.weight;

        if (i != graph.edgeList.size() - 1)
        {
            os << ' ';
        }
        ++i;
    }

    return os;
}
