#pragma once

#include <ATen/core/TensorBody.h>

#include <vector>
#include <set>
#include <ostream>

struct GraphTensorData
{
    at::Tensor x;
    at::Tensor edge_index;
};

class Graph
{
public:
    virtual ~Graph() = default;
    virtual GraphTensorData getAsTensors() = 0;
};


class SizeDelayGraph : public Graph
{
public:
	struct Edge
	{
		unsigned int indexStart;
		unsigned int indexEnd;
		float weight;
	};

    std::vector<Edge> edgeList;
    virtual GraphTensorData getAsTensors() override;

    friend std::ostream& operator<<(std::ostream& os, const SizeDelayGraph& graph);
};

class PacketListGraph : public Graph
{
public:
    std::vector<std::unique_ptr<uint8_t[]>> nodes;
    std::vector<size_t> sizes;
    virtual GraphTensorData getAsTensors() override;
};