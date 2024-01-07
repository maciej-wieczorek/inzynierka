#include "Classifier.h"
#include "Graph.h"

#include <filesystem>

void Classifier::load(std::string modelPath)
{
	try
	{
		std::cout << "Loading model: " << modelPath;
		m_module = torch::jit::load(modelPath);
		m_module.eval();
		std::cout << " Done.\n";
	}
	catch (const std::exception e)
	{
		std::cerr << "\nError loading the model\n";
		std::cerr << e.what();
		std::exit(1);
	}
	catch (...)
	{
		std::cerr << "\nUnknown error loading the model\n";
		std::exit(1);
	}
}

std::vector<float> Classifier::classify(const GraphTensorData& graph)
{
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(graph.x.to(torch::kFloat32) / 255.f);
	inputs.push_back(graph.edge_index);
	auto shape = graph.x.sizes();
	inputs.push_back(torch::full({ shape[0] }, 0, torch::kInt64));
	at::Tensor output = m_module.forward(inputs).toTensor();

	float* data_ptr = output.data_ptr<float>();

	std::vector<float> output_vec(data_ptr, data_ptr + output.numel());

	return output_vec;
}
