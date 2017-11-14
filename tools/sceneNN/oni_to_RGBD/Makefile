all: OniToRGBD.o
	g++ -o OniToRGBD OniToRGBD.o -lfreeimage -lOpenNI -lboost_system -lboost_filesystem

OniToRGBD.o: OniToRGBD.cpp
	g++ -g -c OniToRGBD.cpp -o OniToRGBD.o -I/usr/include/ni/
