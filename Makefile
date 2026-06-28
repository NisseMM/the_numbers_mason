CC = g++-13
CXXFLAGS = -std=c++23 -Wall -Wextra -O3
INCLUDE = -I$(NEURAL_PATH) -Iinclude

run: program
	@./program

program: src/main.cc
	$(CC) -MP -MMD $(CXXFLAGS) $(INCLUDE) $< -o $@

clean:
	@rm program *.d

-include *.d