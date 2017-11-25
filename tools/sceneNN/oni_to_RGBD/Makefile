all: oniToRGBD.o
	g++ -o oniToRGBD oniToRGBD.o -lfreeimage -lOpenNI -lboost_system -lboost_filesystem

oniToRGBD.o: oniToRGBD.cpp
	g++ -g -c oniToRGBD.cpp -o oniToRGBD.o -I/usr/include/ni/
