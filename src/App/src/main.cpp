#include "App.h"

#include <iostream>

int main(int argc, char* argv[])
{
    static const char* usage = "<predict>|<dataset> <capture_file>|<folder_with_captures>|<interface_name> [path_to_model]|[path_to_dataset]";
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << ' ' << usage;
        return 1;
    }

    std::string action = argv[1];
    std::string source = argv[2];
    std::string path = argc == 4 ? argv[3] : "";


    if (action == "dataset" || action == "predict")
    {
		App app{ action, source, path };
        app.run();
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << ' ' << usage;
        return 1;
    }

    return 0;
}

