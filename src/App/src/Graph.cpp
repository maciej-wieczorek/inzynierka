#include <torch/torch.h>

#include "Graph.h"
#include "Grouper.h"

#include <iostream>
#include <fstream>

struct GraphTensorData
{
    torch::Tensor x;
    torch::Tensor edge_index;
};

GraphTensorData graphToTensor(const Graph& graph)
{
    GraphTensorData data;

    std::vector<int64_t> nodes;
    for (const auto& edge : graph.edgeList)
    {
        if (std::find(nodes.begin(), nodes.end(), edge.indexStart) == nodes.end())
        {
            nodes.push_back(edge.indexStart);
        }

        if (std::find(nodes.begin(), nodes.end(), edge.indexEnd) == nodes.end())
        {
            nodes.push_back(edge.indexEnd);
        }
    }

    static std::vector<int64_t> size_to_index(Grouper::numGroups, 0);

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        size_to_index[nodes[i]] = i;
    }

    torch::Tensor indices = torch::tensor(nodes);
    torch::Tensor x_data_1 = torch::one_hot(indices, Grouper::numGroups);

    torch::Tensor edge_pool_1 = torch::zeros({ static_cast<long long>(nodes.size()), Grouper::numGroups });
    torch::Tensor edge_pool_2 = torch::zeros({ static_cast<long long>(nodes.size()), Grouper::numGroups });

    data.edge_index = torch::zeros({ 2, static_cast<long long>(graph.edgeList.size()) }, torch::kInt64);

    for (size_t i = 0; i < graph.edgeList.size(); ++i)
    {
        const auto edge = graph.edgeList[i];

        auto indexStart = size_to_index[edge.indexStart];
        auto indexEnd = size_to_index[edge.indexEnd];

        edge_pool_1[indexStart][indexEnd] = edge.weight;
        edge_pool_2[indexEnd][indexStart] = edge.weight;

        data.edge_index[0][i] = indexStart;
        data.edge_index[1][i] = indexEnd;
    }

    data.x = torch::cat({ x_data_1, edge_pool_1, edge_pool_2 }, 1);


    //std::cout << data.x << '\n';
    //std::cout << data.edge_index << '\n';
    //torch::save({ data.x, data.edge_index }, "tensors.pt");

    return data;
}

std::ostream& operator<<(std::ostream& os, const Graph& graph)
{
    // graphToTensor(graph);
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
