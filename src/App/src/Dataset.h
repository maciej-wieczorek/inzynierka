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

private:
	std::filesystem::path m_dir;

	std::future<void> m_saveFuture;
	std::ofstream m_offsets;
	std::ofstream m_data;
	int64_t m_currentOffset;
};
