#include "App.h"
#include "Splitter.h"

#include <Packet.h>
#include <PcapFileDevice.h>
#include <PcapLiveDeviceList.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <filesystem>

void readCaptureFile(const char* filepath, Splitter* splitter)
{
    std::cout << "Reading: " << filepath << '\n';

   // open a pcap file for reading
    pcpp::PcapFileReaderDevice reader(filepath);
    if (!reader.open())
    {
        std::cerr << "Error opening the capture file: " << filepath << std::endl;
        return;
    }

    // read packet from the file
    pcpp::RawPacket rawPacket;
    while (reader.getNextPacket(rawPacket))
    {
        // parse the raw packet into a parsed packet
        pcpp::Packet parsedPacket(&rawPacket);

        splitter->consumePacket(std::move(parsedPacket));
    }

    // close the file
    reader.close();
}

static bool onPacketArrivesBlockingMode(pcpp::RawPacket* packet, pcpp::PcapLiveDevice* dev, void* cookie)
{
    static size_t packetCounter = 0;
    ++packetCounter;

    // extract the Splitter object form the cookie
    Splitter* splitter = (Splitter*)cookie;

    // parsed the raw packet
    pcpp::Packet parsedPacket(packet);

    splitter->consumePacket(std::move(parsedPacket));

    std::cout << "Packets captured: " << packetCounter << '\r';

    // return false means we don't want to stop capturing after this callback
    return false;
}

void liveCapture(const char* intface, Splitter* splitter)
{
    pcpp::PcapLiveDevice* dev = pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDeviceByIpOrName(intface);

    if (dev)
    {
        if (!dev->open())
        {
            std::cerr << "Cannot open device" << std::endl;
        }
        else
        {
            std::cout << "Starting live capture on: " << intface << "\n";

            // start blocking (main thread) capture with infinite timeout
            dev->startCaptureBlockingMode(onPacketArrivesBlockingMode, splitter, 0);
        }
    }
    else
    {
        std::cerr << "Interface: " << intface << " not found\n";
        std::cout << "Listing available interfaces:\n";

		const std::vector<pcpp::PcapLiveDevice*>& devList = pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDevicesList();
        for (const auto& dev : devList)
        {
            std::cout << "Interface name: " << dev->getName() << "\tInterface description: " << dev->getDesc() << '\n';
        }
    }
}

App::App(std::string action, std::string source, std::string path) :
    m_action{ action }, m_source{ source }, m_path{ path }
{
}

void App::run()
{
    if (m_action == "predict")
    {
        m_webSocketApp.run();
        m_classifier.load(m_path == "" ? "model.pt" : m_path);
    }

	auto splitter = std::make_unique<Splitter>(this);

	if (std::filesystem::is_regular_file(m_source))
	{
		readCaptureFile(m_source.c_str(), splitter.get());
	}
	else if (std::filesystem::is_directory(m_source))
	{
		for (const auto& entry : std::filesystem::directory_iterator(m_source))
		{
			if (entry.is_regular_file())
			{
				readCaptureFile(entry.path().string().c_str(), splitter.get());
			}
		}
	}
	else
	{
		liveCapture(m_source.c_str(), splitter.get());
	}
}

void App::processConnection(const Connection& connection)
{
    GraphTensorData graph = connection.getAsGraph();

    if (m_action == "predict")
    {
        auto output = m_classifier.classify(graph);

        std::stringstream ss;
		ss << connection.m_clinetIP << ':' << connection.m_clientPort << '-' << connection.m_serverIP << ':' << connection.m_serverPort;
        ss << " [";
        for (const auto& elem : output)
        {
            ss << " " << elem;
        }
        ss << ']';

        m_webSocketApp.broadcast(ss.str());
    }
}
