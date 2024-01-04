mkdir deps\release
cd deps\release

if not exist libtorch\ (
    if not exist libtorch-win-shared-with-deps-2.1.2%%2Bcpu.zip curl -O -L "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.2%%2Bcpu.zip"
    powershell -command "Expand-Archive -Path .\libtorch-win-shared-with-deps-2.1.2%%2Bcpu.zip -DestinationPath libtorch"
)

if not exist pcapplusplus\ (
    if not exist pcapplusplus-23.09-windows-vs2022-x64-release.zip curl -O -L https://github.com/seladb/PcapPlusPlus/releases/download/v23.09/pcapplusplus-23.09-windows-vs2022-x64-release.zip
    powershell -command "Expand-Archive -Path .\pcapplusplus-23.09-windows-vs2022-x64-release.zip -DestinationPath pcapplusplus"
)

if not exist npcap-sdk\ (
    if not exist npcap-sdk-1.13.zip curl -O -L https://npcap.com/dist/npcap-sdk-1.13.zip
    powershell -command "Expand-Archive -Path .\npcap-sdk-1.13.zip -DestinationPath npcap-sdk"
)

cd ..\..
cmake -A x64 -G "Visual Studio 17 2022" -S src -B src\build_release -DCMAKE_PREFIX_PATH=%cd%\deps\release\libtorch\libtorch -DPcapPlusPlus_ROOT=%cd%\deps\release\pcapplusplus -DPCAP_ROOT=%cd%\deps\release\npcap-sdk -DPacket_ROOT=%cd%\deps\release\npcap-sdk