#include "App.h"
#include "Splitter.h"
#include "Capture.h"

#include <filesystem>

App::App(std::string action, std::string source, std::string path) :
    m_action{ action }, m_source{ source }, m_path{ path }
{
}

void App::run()
{
    if (m_action == "predict")
    {
        m_webSocketApp.run();
        m_classifier.load(m_path);
    }
	else if (m_action == "dataset")
	{
		m_dataset = std::make_unique<Dataset>();
		m_dataset->open(m_path);
	}


	if (std::filesystem::is_regular_file(m_source))
	{
		auto splitter = std::make_unique<Splitter>(this);
		std::filesystem::path filePath{ m_source };
		splitter->setCurrentDataSource(filePath.filename().string().c_str());
		readCaptureFile(filePath.string().c_str(), splitter.get());
	}
	else if (std::filesystem::is_directory(m_source))
	{
		for (const auto& entry : std::filesystem::directory_iterator(m_source))
		{
			if (entry.is_regular_file())
			{
				auto splitter = std::make_unique<Splitter>(this);
				splitter->setCurrentDataSource(entry.path().filename().string().c_str());
				readCaptureFile(entry.path().string().c_str(), splitter.get());
			}
		}
	}
	else
	{
		auto splitter = std::make_unique<Splitter>(this);
		splitter->setCurrentDataSource("live-capture");
		liveCapture(m_source.c_str(), splitter.get());
	}
}

void App::processConnection(const Connection& connection)
{
    GraphTensorData graph = connection.getAsGraph();
	if (graph.x.sizes()[0] == 0)
	{
		return;
	}

	ConnectionInfo connectionInfo = connection.getConnectionInfo();

    if (m_action == "predict")
    {
        auto output = m_classifier.classify(graph);

        std::stringstream ss;
		ss << connection.m_clientIP << ':' << connection.m_clientPort << '-' << connection.m_serverIP << ':' << connection.m_serverPort;
        ss << " [";
		ss.precision(2);
		ss << std::fixed;
        for (const auto& elem : output)
        {
            ss << " " << elem;
        }
        ss << ']';

        m_webSocketApp.broadcast(ss.str());
    }
	else if (m_action == "dataset")
	{
		m_dataset->add(graph, connectionInfo);
	}
}
