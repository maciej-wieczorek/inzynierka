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
#include "pcapplusplus/IPv4Layer.h"
#include "pcapplusplus/TcpLayer.h"
#include "pcapplusplus/UdpLayer.h"

#include "Splitter.h"

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

        // verify the packet is IPv4
        if (parsedPacket.isPacketOfType(pcpp::IPv4))
        {
            uint16_t srcPort{0}, dstPort{0};

            // extract ports
            if (parsedPacket.isPacketOfType(pcpp::TCP))
            {
                pcpp::TcpLayer* TcpLayer = parsedPacket.getLayerOfType<pcpp::TcpLayer>();
                srcPort = TcpLayer->getSrcPort();
                dstPort = TcpLayer->getDstPort();
            }
            else if (parsedPacket.isPacketOfType(pcpp::UDP))
            {
                pcpp::UdpLayer* UdpLayer = parsedPacket.getLayerOfType<pcpp::UdpLayer>();
                srcPort = UdpLayer->getSrcPort();
                dstPort = UdpLayer->getDstPort();
            }

            // extract IPs
            if (dstPort != 0 && srcPort != 0)
            {
                pcpp::IPv4Address srcIP = parsedPacket.getLayerOfType<pcpp::IPv4Layer>()->getSrcIPv4Address();
                pcpp::IPv4Address dstIP = parsedPacket.getLayerOfType<pcpp::IPv4Layer>()->getDstIPv4Address();

                int packetLen = rawPacket.getRawDataLen();

                // print source and dest IPs
                if (srcPort < dstPort)
                {
                    splitter.add_packet(dstIP, dstPort, srcIP, srcPort, rawPacket.getPacketTimeStamp(), packetLen);
                }
                else
                {
                    splitter.add_packet(srcIP, srcPort, dstIP, dstPort, rawPacket.getPacketTimeStamp(), packetLen);
                }
            }
        }
    }

    // close the file
    reader.close();
}

void liveCapture(const char* intface)
{
    auto& listInstance = pcpp::PcapLiveDeviceList::getInstance();
    pcpp::PcapLiveDevice* dev = pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDeviceByIpOrName(intface);
    const std::vector<pcpp::PcapLiveDevice*>& devList = pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDevicesList();
    // before capturing packets let's print some info about this interface
    std::cout
        << "Interface info:" << std::endl
        << "   Interface name:        " << dev->getName() << std::endl // get interface name
        << "   Interface description: " << dev->getDesc() << std::endl // get interface description
        << "   MAC address:           " << dev->getMacAddress() << std::endl // get interface MAC address
        << "   Default gateway:       " << dev->getDefaultGateway() << std::endl // get default gateway
        << "   Interface MTU:         " << dev->getMtu() << std::endl; // get interface MTU

    if (dev->getDnsServers().size() > 0)
        std::cout << "   DNS server:            " << dev->getDnsServers().at(0) << std::endl;
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

