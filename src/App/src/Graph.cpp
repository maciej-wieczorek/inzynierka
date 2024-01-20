#include <torch/torch.h>

#include "Graph.h"
#include "Grouper.h"

#include <iostream>
#include <fstream>

std::ostream& operator<<(std::ostream& os, const SizeDelayGraph& graph)
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

GraphTensorData SizeDelayGraph::getAsTensors()
{
    GraphTensorData data;

    std::vector<int64_t> nodes;
    for (const auto& edge : edgeList)
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

    std::vector<float> edge_pool_1(nodes.size() * Grouper::numGroups, 0.f);
    std::vector<float> edge_pool_2(nodes.size() * Grouper::numGroups, 0.f);

    std::vector<int64_t> edge_index_data(2 * edgeList.size(), 0);


    for (size_t i = 0; i < edgeList.size(); ++i)
    {
        const auto edge = edgeList[i];

        auto indexStart = size_to_index[edge.indexStart];
        auto indexEnd = size_to_index[edge.indexEnd];

        edge_pool_1[Grouper::numGroups * indexStart + edge.indexEnd] = edge.weight;
        edge_pool_2[Grouper::numGroups * indexEnd + edge.indexStart] = edge.weight;

        edge_index_data[i] = indexStart;
        edge_index_data[edgeList.size() + i] = indexEnd;
    }

    torch::Tensor x_data_2 = torch::from_blob(edge_pool_1.data(), {static_cast<long long>(nodes.size()), Grouper::numGroups}, torch::kFloat32);
    torch::Tensor x_data_3 = torch::from_blob(edge_pool_2.data(), {static_cast<long long>(nodes.size()), Grouper::numGroups}, torch::kFloat32);

    data.x = torch::cat({ x_data_1, x_data_2, x_data_3 }, 1).clone();
    data.edge_index = torch::from_blob(edge_index_data.data(), {2, static_cast<long long>(edgeList.size())}, torch::kInt64).clone();

    return data;
}

GraphTensorData PacketListGraph::getAsTensors()
{
    static constexpr long long featuresSize = 1500;
    torch::Tensor x_data = torch::zeros({ static_cast<long long>(nodes.size()), featuresSize }, torch::kUInt8);
    uint8_t* x_data_ptr = x_data.data_ptr<uint8_t>();

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        memcpy(x_data_ptr + i * featuresSize, nodes[i].get(), std::min(static_cast<long long>(sizes[i]), featuresSize));
    }

    GraphTensorData data;

    //data.x = x_data.to(torch::kFloat32) / 255;
    data.x = x_data;

    std::vector<int64_t> edge_index_data((nodes.size() - 1) * 2, 0);
    for (size_t i = 0; i < nodes.size() - 1; ++i)
    {
        edge_index_data[i] = i;
        edge_index_data[(nodes.size() - 1) + i] = i + 1;
    }

    data.edge_index = torch::from_blob(edge_index_data.data(), { 2, static_cast<long long>(nodes.size()-1) }, torch::kInt64).clone();

    return data;
}
