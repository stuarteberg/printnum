#
# Stupid build script for now; eventually needs proper CMake support
#
# For now, we assume you've already activated a conda environment,
# and that environment has python=2.7, numpy, boost=1.55, and gcc installed into it.

PREFIX=$(python -c "import sys; print sys.prefix")

set -e
mkdir -p build

g++ \
    -shared \
    -fPIC \
    -fno-strict-aliasing \
    -lboost_python \
    -lpython2.7 \
    -I${PREFIX}/include \
    -I${PREFIX}/include/python2.7 \
    -I${PREFIX}/lib/python2.7/site-packages/numpy/core/include \
    -L${PREFIX}/lib \
    -o build/printnum.so \
    src/printnum.cpp \
##

install_name_tool -add_rpath ${PREFIX}/lib build/printnum.so
install_name_tool -change libpython2.7.dylib @rpath/libpython2.7.dylib build/printnum.so
install_name_tool -change libstdc++.6.dylib @rpath/libstdc++.6.dylib build/printnum.so
