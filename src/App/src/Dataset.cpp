#include <torch/torch.h>

#include "Dataset.h"
#include "Graph.h"
#include "Connection.h"
#include "Utilities.h"

#include <regex>

std::string getLabelName(int64_t label)
{
	static const std::vector<std::string> labelNames
	{
		"web", "video-stream", "file-transfer", "chat", "voip", "remote-desktop", "ssh", "other"
	};

	return labelNames.at(label);
}

int64_t extractLabel(std::string fileName)
{
	static const std::vector<std::vector<std::string>> labelMap
	{
		{"web"},
		{"youtube", "netflix", "vimeo"},
		{"scp", "sftp", "rsync"},
		{"chat"},
		{"voip"},
		{"rdp"},
		{"ssh"},
	};

	for (int64_t label = 0; label < labelMap.size(); ++label)
	{
		for (const auto& option : labelMap[label])
		{
			if (containsCaseInsensitive(fileName, option))
			{
				return label;
			}
		}
	};

	return labelMap.size(); // other
}

int findMaxNumberInFilenames(const std::filesystem::path dir)
{
	namespace fs = std::filesystem;
	std::regex filenamePattern("graphs_(\\d+)\\.pt");

	int maxNumber = -1;

	for (const auto& entry : fs::directory_iterator(dir))
	{
		if (fs::is_regular_file(entry))
		{
			std::string filename = entry.path().filename().string();
			std::smatch match;
			if (std::regex_match(filename, match, filenamePattern))
			{
				int currentNumber = std::stoi(match[1].str());
				if (currentNumber > maxNumber)
				{
					maxNumber = currentNumber;
				}
			}
		}
	}

	return maxNumber;
}

size_t getTensorByteSize(const at::Tensor& tensor)
{
	int64_t numel = tensor.numel();
	int64_t itemsize = tensor.element_size();
	return numel * itemsize;
}

Dataset::~Dataset()
{
	save();
	awaitSavedTensors();
}

void Dataset::open(std::string path)
{
	m_dir = path == "" ? "dataset" : path;

	bool dirExists = std::filesystem::exists(m_dir) && std::filesystem::is_directory(m_dir);

	if (dirExists)
	{
		m_saveCount = findMaxNumberInFilenames(m_dir) + 1;
	}
	else
	{
		std::filesystem::create_directory(m_dir);
	}

	std::filesystem::path csvFilePath = m_dir / "graphs.csv";
	m_indexFile.open(csvFilePath, std::ios::app);

	if (std::filesystem::file_size(csvFilePath) == 0)
	{
		// print header
		m_indexFile << "Client IP," << "Client port," << "Server IP," << "Server port,"
			<< "Datasource," << "Count Packets," << "Total Size Packets,"
			<< "First Timestamp," << "Last Timestamp," << "Label," << "Label Name\n";
	}
}

void Dataset::add(const GraphTensorData& graph, const ConnectionInfo& connectionInfo)
{
	int64_t label = extractLabel(connectionInfo.dataSource);
	m_indexFile << connectionInfo << ',' << label << ',' << getLabelName(label) << '\n';
	m_dataToSave.emplace_back(graph.x);
	m_dataToSave.emplace_back(graph.edge_index);
	m_dataToSave.emplace_back(torch::tensor(label));

	m_dataToSaveBytes += getTensorByteSize(graph.x);
	m_dataToSaveBytes += getTensorByteSize(graph.edge_index);
	m_dataToSaveBytes += getTensorByteSize(m_dataToSave[m_dataToSave.size() - 1]);

	++m_numGraphsToSave;

	if (m_dataToSaveBytes >= 10000000)
	{
		save();
	}
}

void Dataset::save()
{
	std::stringstream fileName;
	fileName << "graphs-" << m_numGraphsToSave << '_' << m_saveCount << ".pt";
	std::string filePath = (m_dir / fileName.str()).string();
	saveTensors(m_dataToSave, filePath);
	m_dataToSave.clear();
	m_numGraphsToSave = 0;
	m_dataToSaveBytes = 0;
	++m_saveCount;
}

void Dataset::saveTensors(const std::vector<at::Tensor>& tensors, const std::string& filePath)
{
	std::future<void> future = std::async(std::launch::async, [tensors, filePath]()
	{
		torch::save(tensors, filePath);
	});
	m_saveFutures.push_back(std::move(future));

	if (m_saveFutures.size() >= std::thread::hardware_concurrency())
	{
		awaitSavedTensors();
	}
}

void Dataset::awaitSavedTensors()
{
	for (size_t i = 0; i < m_saveFutures.size(); ++i)
	{
		m_saveFutures[i].get();
	}

	m_saveFutures.clear();
}

