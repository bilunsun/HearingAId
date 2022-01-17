CXX = g++
CXXFLAGS = -g -Wall -Wextra -MMD -std=c++14 -lbluetooth
MAKEFILE_NAME = ${firstword ${MAKEFILE_LIST}}	# makefile name

# object files forming executable
OBJECTS = main.o sound_assist.o audio_processor.o neural_net.o user_out.o
EXEC = soundassist

DEPENDS = ${OBJECTS:.o=.d}						# substitute ".o" with ".d"

.PHONY : all clean

all : ${EXEC}									# build all executables (i.e. just soundassist)

#############################################################

${EXEC} : ${OBJECTS}							# $^ --> expanded ${OBJECTS} and $@ --> expanded ${EXEC}
	${CXX} ${CXXFLAGS} $^ -o $@					

#############################################################

${OBJECTS} : ${MAKEFILE_NAME}					# changes to this file --> recompile

-include ${DEPENDS}								# include *.d files containing program dependences

clean :
	rm -f *.d *.o ${EXEC}
