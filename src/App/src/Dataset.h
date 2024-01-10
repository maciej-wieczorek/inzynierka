#include <ATen/core/TensorBody.h>
#include <string>
#include <filesystem>
#include <fstream>
#include <future>

struct ConnectionInfo;
struct GraphTensorData;


std::string getLabelName(int64_t label);
int64_t extractLabel(std::string fileName);

class Dataset
{
public:
	~Dataset();
	void open(std::string path);
	void add(const GraphTensorData& graph, const ConnectionInfo& connectionInfo);
	void add2(const GraphTensorData& graph, const ConnectionInfo& connectionInfo);
	void save();
	void saveTensors(const std::vector<at::Tensor>& tensors, const std::string& filePath);
	void awaitSavedTensors();

private:
	std::filesystem::path m_dir;
	std::ofstream m_indexFile;
	size_t m_saveCount{};
	std::vector<at::Tensor> m_dataToSave;
	std::vector<std::future<void>> m_saveFutures;
	size_t m_numGraphsToSave{};
	size_t m_dataToSaveBytes{};

	std::ofstream m_offsets;
	std::ofstream m_data;
};
