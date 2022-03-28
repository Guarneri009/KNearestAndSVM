
CC            = g++
#CFLAGS        = -std=c++1y -O0 -g3 -Wall -fmessage-length=0 -Wunused-function -MMD -MP -MF -MT
CFLAGS        = -std=c++14 -O0 -g3 -Wall -fmessage-length=0 -MMD -MP -MF -MT
DEST          = .
LDFLAGS       = -I/usr/include/eigen3
SYMBOLES      = -DCHESSBOARD -DVSCODE
LIBS1         = -lpthread -ldl
LIBS2         = -lopencv_datasets -lopencv_flann -lopencv_xfeatures2d -lopencv_calib3d -lopencv_features2d -lopencv_text 
LIBS3         = -lopencv_videoio -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_ml 
LIBS4         = -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_core
OBJS          = main.cpp
PROGRAM       = main

all:           $(PROGRAM)

$(PROGRAM):    $(OBJS)
	$(CC) $(CFLAGS) $(SYMBOLES) $(LDFLAGS) $(OBJS) -o $(PROGRAM) $(LIBS1) $(LIBS2) $(LIBS3) $(LIBS4)

clean:
	rm -f *.o *~ $(PROGRAM)

docs:
	doxygen doxygen-config
