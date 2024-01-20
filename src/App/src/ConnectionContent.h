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
		Entry(timespec ts, int s, int ps, std::unique_ptr<uint8_t[]>&& packetData) : timestamp{ ts }, size{ s }, payloadSize{ ps }, rawPacketData { std::move(packetData) } {}
		timespec timestamp;
		int size;
		int payloadSize;
		std::unique_ptr<uint8_t[]> rawPacketData;
	};

    size_t getCountPackets() const;
    size_t getTotalSizePackets() const;
    timespec getFirstTimestamp() const;
    timespec getLastTimestamp() const;

	void addEntry(timespec ts, int s, int ps, std::unique_ptr<uint8_t[]>&& packetData);

    size_t m_totalSizePackets{};
	std::vector<Entry> entries;
};

