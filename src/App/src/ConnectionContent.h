#pragma once

#include <IPv4Layer.h>

#include <time.h>
#include <memory>
#include <vector>

class ConnectionContent
{
public:
	struct Entry
	{
		Entry(timespec ts, int s, std::unique_ptr<char[]>&& packetData) : timestamp{ ts }, size{ s }, rawPacketData{ std::move(packetData) } {}
		timespec timestamp;
		int size;
		std::unique_ptr<char[]> rawPacketData;
	};

    size_t getCountPackets() const;
    size_t getTotalSizePackets() const;
    timespec getFirstTimestamp() const;
    timespec getLastTimestamp() const;

	void addEntry(timespec ts, int s, std::unique_ptr<char[]>&& packetData);

    size_t m_totalSizePackets{};
	std::vector<Entry> entries;
};

