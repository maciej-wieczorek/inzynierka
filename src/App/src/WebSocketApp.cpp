#include <uwebsockets/App.h>

#include <time.h>
#include <iostream>
#include "WebSocketApp.h"

uWS::SSLApp* g_app = nullptr;
uWS::Loop* g_loop = nullptr;

void WebSocketApp::run()
{
    struct PerSocketData
    {
    };

	m_thread = std::thread{[]()
	{
		uWS::SSLApp::WebSocketBehavior<PerSocketData> behavior;
		behavior.compression = uWS::SHARED_COMPRESSOR;
		behavior.maxPayloadLength = 16 * 1024 * 1024;
		behavior.idleTimeout = 16;
		behavior.maxBackpressure = 1 * 1024 * 1024;
		behavior.closeOnBackpressureLimit = false;
		behavior.resetIdleTimeoutOnSend = false;
		behavior.sendPingsAutomatically = true;
		behavior.compression = uWS::SHARED_COMPRESSOR;
		behavior.upgrade = nullptr;
		behavior.open = [](auto* ws)
		{
			ws->subscribe("broadcast");
		};
		behavior.message = [](auto*/*ws*/, std::string_view /*message*/, uWS::OpCode /*opCode*/)
		{
		};
		behavior.drain = [](auto*/*ws*/)
		{
		};
		behavior.ping = [](auto*/*ws*/, std::string_view)
		{
		};
		behavior.pong = [](auto*/*ws*/, std::string_view)
		{
		};
		behavior.close = [](auto*/*ws*/, int /*code*/, std::string_view /*message*/)
		{
		};

		uWS::SocketContextOptions options{};

		uWS::SSLApp app = uWS::SSLApp(options);
		app.ws<PerSocketData>("/*", std::move(behavior)).listen(9001, [](auto* listen_socket)
		{
			if (listen_socket)
			{
				std::cout << "Listening on port " << 9001 << std::endl;
			}
		});

		g_loop = uWS::Loop::get();
		g_app = &app;

		app.run();
	}};

	while (!g_app || !g_loop)
	{
		std::this_thread::sleep_for(std::chrono::microseconds(10));
	}
}

void WebSocketApp::broadcast(const std::string& data)
{
	g_loop->defer([data]() {g_app->publish("broadcast", data, uWS::OpCode::TEXT); });
}
