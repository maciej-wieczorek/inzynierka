#include <torch/torch.h>

#include "Dataset.h"
#include "Graph.h"
#include "Connection.h"
#include "Utilities.h"

#include <regex>

std::string getLabelNameVNAT(int64_t label)
{
	static const std::vector<std::string> labelNames
	{
		"nonvpn-netflix", "nonvpn-rdp", "nonvpn-rsync", "nonvpn-scp", "nonvpn-sftp", "nonvpn-skype-chat", "nonvpn-ssh", "nonvpn-vimeo", "nonvpn-voip", "nonvpn-youtube",
		"vpn-netflix", "vpn-rdp", "vpn-rsync", "vpn-scp", "vpn-sftp", "vpn-skype-chat", "vpn-ssh", "vpn-vimeo", "vpn-voip", "vpn-youtube",
	};

	return labelNames.at(label);
}

int64_t extractLabelVNAT(std::string fileName)
{
	static const std::vector<std::vector<std::string>> labelMap
	{
		{"nonvpn_netflix"}, {"nonvpn_rdp"}, {"nonvpn_rsync"}, {"nonvpn_scp"}, {"nonvpn_sftp"}, {"nonvpn_skype"}, {"nonvpn_ssh"}, {"nonvpn_vimeo"}, {"nonvpn_voip"}, {"nonvpn_youtube"},
		{"vpn_netflix"}, {"vpn_rdp"}, {"vpn_rsync"}, {"vpn_scp"}, {"vpn_sftp"}, {"vpn_skype"}, {"vpn_ssh"}, {"vpn_vimeo"}, {"vpn_voip"}, {"vpn_youtube"},
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

	std::cerr << "Label not found!\n";
	return labelMap.size(); // other
}

std::string getLabelNameMalware(int64_t label)
{
	static const std::vector<std::string> labelNames
	{
		"benign", "malware"
	};

	return labelNames.at(label);
}

int64_t extractLabelMalware(std::string fileName)
{
	static const std::vector<std::vector<std::string>> labelMap
	{
		{"benign"}, {"malware"}
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

	std::cerr << "Label not found!\n";
	return labelMap.size(); // other
}

std::string getLabelNamePMNK(int64_t label)
{
	static const std::vector<std::string> labelNames
	{
		"bulk", "idle", "web", "video", "interactive"
	};

	return labelNames.at(label);
}

int64_t extractLabelPMNK(std::string fileName)
{
	static const std::vector<std::vector<std::string>> labelMap
	{
		{"bulk"}, {"idle"}, {"web"}, {"video"}, {"interactive"}
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

	std::cerr << "Label not found!\n";
	return labelMap.size(); // other
}

std::string getLabelName(int64_t label)
{
	static const std::vector<std::string> labelNames
	{
		"idle", "web", "video", "voip", "file-transfer", "online-game"
	};

	return labelNames.at(label);
}

int64_t extractLabel(std::string fileName)
{
	static const std::vector<std::vector<std::string>> labelMap
	{
		{"idle"}, {"web"}, {"video"}, {"voip"}, {"file-transfer"}, {"game"}
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

	std::cerr << "Label not found!\n";
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
}

void Dataset::open(std::string path)
{
	m_dir = path;

	bool dirExists = std::filesystem::exists(m_dir) && std::filesystem::is_directory(m_dir);

	if (!dirExists)
	{
		std::filesystem::create_directory(m_dir);
	}


	m_data.open(m_dir / "data.bin", std::ios::binary | std::ios::app);
	m_offsets.open(m_dir / "offsets.bin", std::ios::binary | std::ios::app);
	if (std::filesystem::is_regular_file(m_dir / "data.bin"))
	{
		m_currentOffset = std::filesystem::file_size(m_dir / "data.bin");
	}
	else
	{
		m_currentOffset = 0;
	}
}

void Dataset::add(const GraphTensorData& graph, const ConnectionInfo& connectionInfo)
{
	if (m_saveFuture.valid())
	{
		m_saveFuture.get();
	}

	m_saveFuture = std::async(std::launch::async, [graph, connectionInfo, this]()
	{
		int64_t label = extractLabel(connectionInfo.dataSource);

		m_offsets.write(reinterpret_cast<const char*>(&m_currentOffset), sizeof(m_currentOffset));
		m_offsets.write(reinterpret_cast<const char*>(&label), sizeof(label));

		auto x_data = graph.x.data_ptr();
		auto edge_index_data = graph.edge_index.data_ptr();

		size_t x_data_size = getTensorByteSize(graph.x);
		auto x_shape = graph.x.sizes();
		size_t edge_index_data_size = getTensorByteSize(graph.edge_index);

		int64_t x_data_shape_n = x_shape[0];
		int64_t x_data_shape_m = x_shape[1];
		int64_t x_data_type = graph.x.dtype() == torch::kFloat32 ? 0 : 1;
		int64_t edge_index_size = graph.edge_index.numel();

		// write data info
		m_data.write(reinterpret_cast<const char*>(&x_data_shape_n), sizeof(x_data_shape_n));
		m_data.write(reinterpret_cast<const char*>(&x_data_shape_m), sizeof(x_data_shape_m));
		m_data.write(reinterpret_cast<const char*>(&x_data_type), sizeof(x_data_type));
		m_data.write(reinterpret_cast<const char*>(&edge_index_size), sizeof(edge_index_size));

		// write data
		m_data.write(reinterpret_cast<const char*>(x_data), x_data_size);
		m_data.write(reinterpret_cast<const char*>(edge_index_data), edge_index_data_size);
		m_data.write(reinterpret_cast<const char*>(&label), sizeof(label));

		m_currentOffset += 4 * sizeof(int64_t) + x_data_size + edge_index_data_size + sizeof(label);
	});
}

