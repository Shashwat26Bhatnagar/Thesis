# fastodeint ![](https://img.shields.io/badge/python-3.6.8-orange) ![](https://img.shields.io/badge/c%2B%2B-11-blue)

An ODE solver C++ implementation just for PenSimPy

## Prerequisites

* A compiler with C++11 support
* CMake >= 2.8.12
* boost
* openMP
* python 3.6.8 or greater

**On macOS**

If you already have had Xcode installed, then you can skip the first line below
```bash
xcode-select --install
brew install cmake
brew install boost
brew install libomp
```

**On Linux (Ubuntu)**
```bash
sudo apt-get install libomp-dev
sudo apt-get install libboost-all-dev
```
If you on a different linux distribution, please find equivalent ways to install those dependencies.

**On Windows 10**

Please make sure that you have Microsoft Visual Studio 2019 C++ installed. One way to verify it is to 
use its uninstaller in "Apps & features". 

![](https://github.com/Quarticai/fastodeint/blob/master/img/uninstaller.png)

Click on "modify" button, then you will be able to see the components

![](https://github.com/Quarticai/fastodeint/blob/master/img/vs-c%2B%2B-install.png)

"Desktop development with C++" must be checked. Also, "C++ CMake tools for windows" must be checked in
optional section

If you don't have one, then you can get a free copy of Visual Studio 2019 Community from 
https://visualstudio.microsoft.com/downloads/

Like MacOS, Windows doesn't ship with a package manager, so here we chose `vcpkg`, a C++ Library Manger developed by Microsoft
to help us with `boost` installation on Windows. https://github.com/microsoft/vcpkg provides detailed steps on
how to install `vcpkg`.

Suppose you've downloaded `vcpkg` and it's located at `C:\vcpkg`, you need to bootstrap it at very first time. Open up
Command Prompt with admin privileges and run the following commands 
```bash
cd C:\vcpkg
.\bootstrap-vcpkg.bat
```
Upon successful completion of bootstrapping, you will be able to see `vcpkg.exe` in `C:\vcpkg`

You are all set for `boost` installation! Now, run the following command to get `boost` 
(Note that this step may take a long time to run)
```bash
vcpkg install boost:x86-windows-static 
```
Next, we need to find the file that can tell `CMake` where to find all C++ libraries we downloaded using
`vcpkg.exe`. You should be able to find it at `PATH_TO_VCPKG\scripts\buildsystems\vcpkg.cmake`, where `PATH_TO_VCPKG` is the location
of `vcpkg` directory. In our case, the full path would look like the following

```bash
C:\vcpkg\scripts\buildsystems\vcpkg.cmake
```
Keep the path in a notepad or somewhere else, as you will need it when installing `fastodeint`


## Installation

**On Mac/Ubuntu**

You could either do `pip install fastodeint` or clone this repository and pip install. 
For the later one,  the `--recursive` option which is needed for the pybind11 submodule:

```bash
git clone --recursive https://github.com/Quarticai/fastodeint.git
pip install ./fastodeint
```

**On Windows**

First, you need to run "x86 Native Tools Command Prompt for VS 2019" as administrator

![](https://github.com/Quarticai/fastodeint/blob/master/img/vs-x86-cmd.png)

Next, create an environment variable `CMAKE_TOOLCHAIN_FILE` with `PATH_TO_VCPKG\scripts\buildsystems\vcpkg.cmake`
as the value. In our case, the command would look like the following:
```bash
set CMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
```
In the same Command Prompt, install `fastodeint` using pip
```bash
python -m pip install fastodeint
```

*Troubleshooting Tips (For Windows):*
```bash
CMake Error at pybind11/tools/FindPythonLibsNew.cmake:122 (message):
 Python config failure: Python is 32-bit, chosen compiler is 64-bit
```
That means your Python only supports 32-bit,
and the solution is to run the command above with a Python compiler that is x86-64 mode compatible. If you don't have one, 
you can download the ones with `Windows x86-64` in their names from https://www.python.org/downloads/windows/

```bash
Could NOT find Boost (missing: Boost_INCLUDE_DIR)
```
Double check the environment variable `CMAKE_TOOLCHAIN_FILE`. It's possible that you forgot to set environment
variable or set an invalid path.

## License

fastodeint is provided under a BSD-style license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.


## Example

```python
import fastodeint
fastodeint.integrate(initial_state, params, start_time, end_time, dt)
```
