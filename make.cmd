@echo off
if "%1" == "all" (
    call %0 install
    call %0 check
    goto :eof
)
if "%1" == "check" (
    echo ***
    echo *** Running test suite
    echo ***
    python -m unittest tests -v
    goto :eof
)
if "%1" == "clean" (
    echo ***
    echo *** Cleaning files
    echo ***
    for /D /r %%i in (__pycache*__) do rmdir /S /Q "%%i"
    for /D %%i in (build dist iq.egg-info) do if exist "%%i" rmdir /S /Q "%%i"
    goto :eof
)
if "%1" == "install" (
    call %0 clean
    call %0 reinstall
    goto :eof
)
if "%1" == "reinstall" (
    echo ***
    echo *** Installing library
    echo ***
    pip uninstall -y superqulan
    pip install .
    goto :eof
)
echo Unknown option %1
exit -1
if "%1" == "docs" (
    cd docs
    call %0 html
    cd ..
)
