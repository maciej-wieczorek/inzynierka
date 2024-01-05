#include <torch/script.h>

#include <vector>
#include <string>

struct GraphTensorData;

class Classifier
{
public:
	void load(std::string modelPath);
	std::vector<float> classify(const GraphTensorData& graph);

private:
	torch::jit::script::Module m_module;
};
