#pragma once

#include <vector>
#include <set>
#include <ostream>

struct Edge
{
    unsigned int indexStart;
    unsigned int indexEnd;
    float weight;
};

struct Graph
{
    std::vector<Edge> edgeList;

    friend std::ostream& operator<<(std::ostream& os, const Graph& graph);
};