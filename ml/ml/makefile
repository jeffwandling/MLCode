## makefile for non-windows platforms

CXX = g++

CXXFLAGS :=  -O2 -D_DEBUG -Iinc

EXE := prog
OBJ := main.o Data.o DataHandler.o Knn.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

prog: $(OBJ)
	$(CXX) $^ -o $@

clean:
	rm -f $(OBJ) $(EXE)
