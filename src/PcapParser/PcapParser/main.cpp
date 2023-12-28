#include "Defines.h"

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <filesystem>

#include "pcapplusplus/Packet.h"
#include "pcapplusplus/PcapFileDevice.h"
#include "pcapplusplus/PcapLiveDeviceList.h"

#include "Splitter.h"

#include <windows.h>

void readCaptureFile(const char* filepath)
{
    std::cout << "Reading: " << filepath << '\n';

    Splitter splitter{ filepath };

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

        splitter.consumePacket(parsedPacket);
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

    splitter->consumePacket(parsedPacket);

    std::cout << "Packets captured: " << packetCounter << '\r';

    // return false means we don't want to stop capturing after this callback
    return false;
}

void liveCapture(const char* intface)
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
            auto splitter = std::make_unique<Splitter>("live-capture");

            // start blocking (main thread) capture with infinite timeout
            dev->startCaptureBlockingMode(onPacketArrivesBlockingMode, splitter.get(), 0);
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

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <capture_file or folder_with_captures or interface_name>";
        return 1;
    }

    if (std::filesystem::is_regular_file(argv[1]))
    {
        readCaptureFile(argv[1]);
    }
    else if (std::filesystem::is_directory(argv[1]))
    {
        for (const auto& entry : std::filesystem::directory_iterator(argv[1]))
        {
            if (entry.is_regular_file())
            {
                readCaptureFile(entry.path().string().c_str());
            }
        }
    }
    else
    {
        liveCapture(argv[1]);
    }

    return 0;
}

