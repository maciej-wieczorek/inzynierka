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
    std::vector<const char*> nodes;
    virtual GraphTensorData getAsTensors() override;
};