NVCC := nvcc
CXXFLAGS := -O3 -std=c++11
NVCCFLAGS := -O3 -arch=sm_50
LIBS := -lsndfile -lcufft

TARGET := bin/praanai
SRCDIR := src
OBJDIR := obj

SOURCES := $(wildcard $(SRCDIR)/*.cu)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

$(TARGET): $(OBJECTS) | bin
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

bin:
	mkdir -p bin

clean:
	rm -rf $(OBJDIR) bin

.PHONY: clean