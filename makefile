#Compiler and Linker
CC          := g++
NVCC        := nvcc

#The Target Binary Program
TARGET_S      := test
TARGET_P      := parallel_test

#The Directories, Source, Includes, Objects, Binary and Resources
BUILDDIR_S  := ./build/serial
BUILDDIR_P  := ./build/parallel
TARGETDIR   := .
SRCEXT      := cpp
DEPEXT      := d
OBJEXT      := o

COREPATH    := src/core
UIPATH      := src/ui
BUILDDIR_UI := build/ui
UILIBPATH   := $(BUILDDIR_UI)/visualizer.a

#Flags, Libraries and Includes
CCFLAGS      := -w -std=c++11 -pthread -I$(COREPATH) -I$(UIPATH)
NVCCFLAGS    := -w -std=c++11 -Wno-deprecated-gpu-targets -x cu -DPARALLEL -I$(COREPATH) -I$(UIPATH)
NVCCLINK     := -Wno-deprecated-gpu-targets -L/usr/local/cuda-8.0/lib64 -DPARALLEL -lcuda -lcudart
LIBS         := `pkg-config --libs gtkmm-3.0`

#Default Make
all: serial

#---------------------------------------------------------------------------------
#  UI BUILDING
#---------------------------------------------------------------------------------

$(UILIBPATH): $(BUILDDIR_UI)/visualizer.o $(BUILDDIR_UI)/gui.o
	ar rvs $(UILIBPATH) $(BUILDDIR_UI)/visualizer.o $(BUILDDIR_UI)/gui.o

SOURCES_UI    := $(shell find $(UIPATH) -type f -name *.$(SRCEXT))
OBJECTS_UI    := $(patsubst $(UIPATH)/%,$(BUILDDIR_UI)/%,$(SOURCES_UI:.$(SRCEXT)=.$(OBJEXT)))

#Pull in dependency info for *existing* .o files
-include $(OBJECTS_UI:.$(OBJEXT)=.$(DEPEXT))

#Link
$(TARGET_UI): $(UILIBPATH) $(OBJECTS_UI)
	$(CC) $(CCFLAGS) -I$(COREPATH) -I$(UIPATH) `pkg-config --cflags gtkmm-3.0` -c $^ -o $(BUILDDIR_UI)/$(TARGET_UI) $(LIBS)

#Compile
$(BUILDDIR_UI)/%.$(OBJEXT): $(UIPATH)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -I$(COREPATH) -I$(UIPATH) `pkg-config --cflags gtkmm-3.0` -c -o $@ $< $(LIBS)
	@$(CC) $(CCFLAGS) -MM $(UIPATH)/$*.$(SRCEXT) > $(BUILDDIR_UI)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR_UI)/$*.$(DEPEXT) $(BUILDDIR_UI)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR_UI)/$*.$(OBJEXT):|' < $(BUILDDIR_UI)/$*.$(DEPEXT).tmp > $(BUILDDIR_UI)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR_UI)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR_UI)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR_UI)/$*.$(DEPEXT).tmp

#---------------------------------------------------------------------------------
#  CORE BUILDING
#---------------------------------------------------------------------------------

#------------- SERIAL ------------------------
SOURCES       := $(shell find $(COREPATH) -type f -name *.$(SRCEXT))
OBJECTS_S     := $(patsubst $(COREPATH)/%,$(BUILDDIR_S)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

serial: directories $(TARGET_S)

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
$(BUILDDIR_S)/%.$(OBJEXT): $(COREPATH)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c -o $@ $<
	@$(CC) $(CCFLAGS) -MM $(COREPATH)/$*.$(SRCEXT) > $(BUILDDIR_S)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR_S)/$*.$(DEPEXT) $(BUILDDIR_S)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR_S)/$*.$(OBJEXT):|' < $(BUILDDIR_S)/$*.$(DEPEXT).tmp > $(BUILDDIR_S)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR_S)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR_S)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR_S)/$*.$(DEPEXT).tmp

#------------- PARALLEL ------------------------
SOURCES       := $(shell find $(COREPATH) -type f -name *.$(SRCEXT))
OBJECTS_P     := $(patsubst $(COREPATH)/%,$(BUILDDIR_P)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

parallel: directories $(TARGET_P)

#Pull in dependency info for *existing* .o files
-include $(OBJECTS_P:.$(OBJEXT)=.$(DEPEXT))

$(TARGET_P): $(UILIBPATH) $(OBJECTS_P)
	$(NVCC) $(NVCCLINK) -o $(TARGETDIR)/$(TARGET_P) $^ $(UILIBPATH) $(LIBS)

$(BUILDDIR_P)/%.$(OBJEXT): $(COREPATH)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<
	@$(NVCC) $(NVCCFLAGS) -M $(COREPATH)/$*.$(SRCEXT) > $(BUILDDIR_P)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR_P)/$*.$(DEPEXT) $(BUILDDIR_P)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR_P)/$*.$(OBJEXT):|' < $(BUILDDIR_P)/$*.$(DEPEXT).tmp > $(BUILDDIR_P)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR_P)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR_P)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR_P)/$*.$(DEPEXT).tmp

#Non-File Targets
.PHONY: all remake clean
