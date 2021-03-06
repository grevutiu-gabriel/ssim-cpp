## ssim-cpp

An alternative written in C++ for pyssim ( A Python module for computing the Structural Similarity Image Metric (SSIM)) from https://github.com/jterrace/pyssim

In order to compile run in Terminal:

h5c++ ssim-cpp.cpp -o ssim-cpp -I/usr/include/opencv2 `pkg-config --libs --cflags opencv` -larmadillo -std=gnu++11

or

g++ ssim-cpp.cpp -o ssim-cpp -I/usr/include/hdf5/serial -I/usr/include/opencv2 `pkg-config --libs --cflags opencv` -larmadillo -std=gnu++11

or

g++ ssim-cpp.cpp -o ssim-cpp -I/usr/include/hdf5/serial -Ofast -march=native -I/usr/include/opencv2 `pkg-config --libs --cflags opencv` -larmadillo -std=gnu++11

or

g++ ssim-cpp.cpp -o ssim-cpp -I/usr/include/hdf5/serial -Ofast -march=silvermont -I/usr/include/opencv2 `pkg-config --libs --cflags opencv` -larmadillo -std=gnu++11

or

g++ ssim-cpp.cpp -o ssim-cpp -I/usr/include/hdf5/serial -Ofast -march=silvermont -I/usr/include/opencv2 `pkg-config --libs --cflags opencv` -larmadillo -std=gnu++11 -fopenmp (in order to call OpenMP API)

To execute the program run:

./ssim-cpp img1.jpg img2.jpg

or

./ssim-cpp listofimages.txt (2 filenames per line, separated by one space)

To obtain source code run in Terminal:

git clone https://github.com/grevutiu-gabriel/ssim-cpp.git
