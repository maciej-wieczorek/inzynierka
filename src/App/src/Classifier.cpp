#include "Classifier.h"
#include "Graph.h"

void Classifier::load(std::string modelPath)
{
	m_module = torch::jit::load(modelPath);
}

std::vector<float> Classifier::classify(const GraphTensorData& graph)
{
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(graph.x);
	inputs.push_back(graph.edge_index);
	auto shape = graph.x.sizes();
	inputs.push_back(torch::full({ shape[0] }, 0, torch::kInt64));
	at::Tensor output = m_module.forward(inputs).toTensor();

	float* data_ptr = output.data_ptr<float>();

	std::vector<float> result(data_ptr, data_ptr + output.numel());

	return result;
}
