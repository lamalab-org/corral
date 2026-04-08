SET VCPKG_ROOT=%CD%\vcpkg

SET VCPKG_BUILD_TYPE=release

:: Run vcpkg bootstrap script
CALL %VCPKG_ROOT%\bootstrap-vcpkg.bat
if %errorlevel% NEQ 0 exit /b %errorlevel%

SET VCPKG_EXE=%VCPKG_ROOT%\vcpkg.exe

%VCPKG_EXE% install "libarchive[bzip2,lz4,lzma,lzo,crypto,zstd]" --triplet x64-windows-static-md
if %errorlevel% NEQ 0 exit /b %errorlevel%
%VCPKG_EXE% install "curl" --triplet x64-windows-static-md
if %errorlevel% NEQ 0 exit /b %errorlevel%
%VCPKG_EXE% install "libiconv" --triplet x64-windows-static-md
if %errorlevel% NEQ 0 exit /b %errorlevel%
%VCPKG_EXE% install "libxml2" --triplet x64-windows-static-md
if %errorlevel% NEQ 0 exit /b %errorlevel%

SET "CXXFLAGS=%CXXFLAGS% /showIncludes"
SET "CXXFLAGS=%CXXFLAGS% /D YAML_CPP_STATIC_DEFINE"
SET CMAKE_PREFIX_PATH=%VCPKG_ROOT%\installed\x64-windows-static-md\;%CMAKE_PREFIX_PATH%

cmake -S mamba ^
    -B build ^
    -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDLL" ^
    -D CMAKE_INSTALL_PREFIX=%LIBRARY_PREFIX% ^
    -D CMAKE_PREFIX_PATH="%VCPKG_ROOT%\installed\x64-windows-static-md\;%CMAKE_PREFIX_PATH%" ^
    -D CMAKE_BUILD_TYPE="Release" ^
    -D BUILD_LIBMAMBA=ON ^
    -D BUILD_LIBMAMBA_SPDLOG=ON ^
    -D BUILD_STATIC=ON ^
    -D BUILD_MICROMAMBA=ON ^
    -G "Ninja"
if %errorlevel% NEQ 0 exit /b %errorlevel%

cmake --build build --parallel %CPU_COUNT% --verbose
if %errorlevel% NEQ 0 exit /b %errorlevel%

cmake --install build
if %errorlevel% NEQ 0 exit /b %errorlevel%

DEL /Q /F /S "%LIBRARY_PREFIX%\lib\libmamba*"
if %errorlevel% NEQ 0 exit /b %errorlevel%
RMDIR /S /Q "%LIBRARY_PREFIX%\include\mamba"
if %errorlevel% NEQ 0 exit /b %errorlevel%
RMDIR /S /Q "%LIBRARY_PREFIX%\lib\cmake\libmamba"
if %errorlevel% NEQ 0 exit /b %errorlevel%
