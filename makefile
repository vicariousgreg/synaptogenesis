#Default Make
all: directories serial

COREPATH    := src/core
UIPATH      := src/ui
BUILDDIR_UI := build/ui
UILIBPATH   := $(BUILDDIR_UI)/visualizer.a

#---------------------------------------------------------------------------------
#  UI BUILDING
#---------------------------------------------------------------------------------

$(UILIBPATH): $(BUILDDIR_UI)/visualizer.o $(BUILDDIR_UI)/gui.o
	ar rvs $(UILIBPATH) $(BUILDDIR_UI)/visualizer.o $(BUILDDIR_UI)/gui.o

$(BUILDDIR_UI)/visualizer.o: $(UIPATH)/visualizer.cpp $(UIPATH)/visualizer.h $(UIPATH)/gui.h $(UIPATH)/layer_info.h
	g++ -I$(COREPATH) -I$(UIPATH) `pkg-config --cflags gtkmm-3.0` -c $(UIPATH)/visualizer.cpp -o $(BUILDDIR_UI)/visualizer.o $(LIBS)

$(BUILDDIR_UI)/gui.o: $(UIPATH)/gui.cpp $(UIPATH)/gui.h $(UIPATH)/layer_info.h
	g++ -I$(COREPATH) -I$(UIPATH) `pkg-config --cflags gtkmm-3.0` -c $(UIPATH)/gui.cpp -o $(BUILDDIR_UI)/gui.o $(LIBS)

#---------------------------------------------------------------------------------
#  CORE BUILDING
#---------------------------------------------------------------------------------

#Compiler and Linker
CC          := g++
NVCC        := nvcc

#The Target Binary Program
TARGET_S      := test
TARGET_P      := parallel_test

#The Directories, Source, Includes, Objects, Binary and Resources
SRCDIR      := $(COREPATH)
BUILDDIR_S  := ./build/serial
BUILDDIR_P  := ./build/parallel
TARGETDIR   := .
SRCEXT      := cpp
DEPEXT      := d
OBJEXT      := o

#Flags, Libraries and Includes
CCFLAGS      := -w -std=c++11 -pthread -I$(COREPATH) -I$(UIPATH)
NVCCFLAGS    := -w -std=c++11 -Wno-deprecated-gpu-targets -x cu -DPARALLEL -I$(COREPATH) -I$(UIPATH)
NVCCLINK     := -Wno-deprecated-gpu-targets -L/usr/local/cuda-8.0/lib64 -DPARALLEL -lcuda -lcudart
LIBS         := `pkg-config --libs gtkmm-3.0`

#------------- SERIAL ------------------------
SOURCES       := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS_S     := $(patsubst $(SRCDIR)/%,$(BUILDDIR_S)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

serial: $(TARGET_S)

#Make the Directories
directories:
	@mkdir -p $(BUILDDIR_S)
	@mkdir -p $(BUILDDIR_P)
	@mkdir -p $(BUILDDIR_UI)

#Clean only Objects
clean:
	@$(RM) -rf $(BUILDDIR_S)
	@$(RM) -rf $(BUILDDIR_P)
	@$(RM) -rf $(BUILDDIR_UI)

#Pull in dependency info for *existing* .o files
-include $(OBJECTS_S:.$(OBJEXT)=.$(DEPEXT))

#Link
$(TARGET_S): $(UILIBPATH) $(OBJECTS_S)
	$(CC) $(CCFLAGS) -o $(TARGETDIR)/$(TARGET_S) $^ $(UILIBPATH) $(LIBS)

#Compile
$(BUILDDIR_S)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c -o $@ $<
	@$(CC) $(CCFLAGS) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR_S)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR_S)/$*.$(DEPEXT) $(BUILDDIR_S)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR_S)/$*.$(OBJEXT):|' < $(BUILDDIR_S)/$*.$(DEPEXT).tmp > $(BUILDDIR_S)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR_S)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR_S)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR_S)/$*.$(DEPEXT).tmp

#------------- PARALLEL ------------------------
SOURCES       := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS_P     := $(patsubst $(SRCDIR)/%,$(BUILDDIR_P)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

parallel: $(TARGET_P)

#Pull in dependency info for *existing* .o files
-include $(OBJECTS_P:.$(OBJEXT)=.$(DEPEXT))

$(TARGET_P): $(UILIBPATH) $(OBJECTS_P)
	$(NVCC) $(NVCCLINK) -o $(TARGETDIR)/$(TARGET_P) $^ $(UILIBPATH) $(LIBS)

$(BUILDDIR_P)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<
	@$(NVCC) $(NVCCFLAGS) -M $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR_P)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR_P)/$*.$(DEPEXT) $(BUILDDIR_P)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR_P)/$*.$(OBJEXT):|' < $(BUILDDIR_P)/$*.$(DEPEXT).tmp > $(BUILDDIR_P)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR_P)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR_P)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR_P)/$*.$(DEPEXT).tmp

#Non-File Targets
.PHONY: all remake clean
