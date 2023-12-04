#include "ConnectionContent.h"

ConnectionEntry::ConnectionEntry(timespec ts, int s)
    : timestamp{ ts }, size{ s }
{
}

