CTAGS  := $(shell command -v ctags 2> /dev/null)
CSCOPE := $(shell command -v cscope 2> /dev/null)

ifndef JOBS
JOBS:=1
OS:=$(shell uname -s)

ifeq ($(OS),Linux)
	JOBS:=$(shell grep -c ^processor /proc/cpuinfo)
endif
ifeq ($(OS),Darwin) # Assume Mac OS X
	JOBS:=$(shell system_profiler | awk '/Number Of CPUs/{print $4}{next;}')
endif
endif

MAKEFLAGS += --jobs=$(JOBS)

#Compiler and Linker
CC            := g++
NVCC          := /usr/local/cuda-8.0/bin/nvcc

#The Target Binary Program
TARGET_S      := syngen_test
TARGET_P      := syngen_parallel_test
LIBRARY       := libsyngen.so

#The Directories, Source, Includes, Objects, Binary and Resources
BUILDDIR_LINK := ./build
BUILDDIR_S    := ./build/serial
BUILDDIR_P    := ./build/parallel
LIBRARY_DIR   := ./lib
BIN_DIR       := ./bin
SRCEXT        := cpp
DEPEXT        := d
OBJEXT        := o

COREPATH      := src/core
UIPATH        := src/ui
LIBSPATH      := src/libs
BUILDDIR_UI   := build/ui
BUILDDIR_LIBS := build/libs
UILIBPATH     := $(BUILDDIR_UI)/gui.a

#Flags, Libraries and Includes
CCFLAGS      := -w -fPIC -std=c++11 -pthread -I$(COREPATH) -I$(UIPATH) -I$(LIBSPATH)
NVCCFLAGS    := -w -Xcompiler -fPIC -std=c++11 -Wno-deprecated-gpu-targets -x cu -I$(COREPATH) -I$(UIPATH) -I$(LIBSPATH)
NVCCLINK     := -w -Wno-deprecated-gpu-targets -L/usr/local/cuda-8.0/lib64 -lcuda -lcudart
LIBS         := `pkg-config --libs gtkmm-3.0`

ifndef MAIN
TARGET_S=
TARGET_P=
endif

ifdef DEBUG
CCFLAGS+=-g
NVCCFLAGS+=-g
CCFLAGS+=-DDEBUG
NVCCFLAGS+=-DDEBUG
endif

ifndef NO_GUI
CCFLAGS+=-D__GUI__
NVCCFLAGS+=-D__GUI__
else
UILIBPATH=
LIBS=
endif

#Default Make
all: serial

install: 
	python install_python.py
	sudo cp $(LIBRARY_DIR)/$(LIBRARY) /usr/lib/

#---------------------------------------------------------------------------------
#  LIBS BUILDING
#---------------------------------------------------------------------------------

SOURCES_LIBS    := $(shell find $(LIBSPATH) -type f -name *.$(SRCEXT))
OBJECTS_LIBS    := $(patsubst $(LIBSPATH)/%,$(BUILDDIR_LIBS)/%,$(SOURCES_LIBS:.$(SRCEXT)=.$(OBJEXT)))

libs: directories $(OBJECTS_LIBS)

$(BUILDDIR_LIBS)/%.$(OBJEXT): $(LIBSPATH)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c -o $@ $<

#---------------------------------------------------------------------------------
#  UI BUILDING
#---------------------------------------------------------------------------------

SOURCES_UI    := $(shell find $(UIPATH) -type f -name *.$(SRCEXT))
OBJECTS_UI    := $(patsubst $(UIPATH)/%,$(BUILDDIR_UI)/%,$(SOURCES_UI:.$(SRCEXT)=.$(OBJEXT)))

$(UILIBPATH): $(OBJECTS_UI)
	ar rvs $(UILIBPATH) $^

#Pull in dependency info for *existing* .o files
-include $(OBJECTS_UI:.$(OBJEXT)=.$(DEPEXT))

#Compile
$(BUILDDIR_UI)/%.$(OBJEXT): $(UIPATH)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) `pkg-config --cflags gtkmm-3.0` -c -o $@ $< $(LIBS)
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

serial: directories libs $(UILIBPATH) $(OBJECTS_S) $(TARGET_S) $(OBJECTS_LIBS) ctags_s
	$(CC) $(CCFLAGS) -shared -o $(LIBRARY_DIR)/$(LIBRARY) $(OBJECTS_S) $(OBJECTS_LIBS) $(UILIBPATH) $(LIBS)

#Make tags
ctags_s: $(OBJECTS_S)
ifdef CTAGS
	ctags -R --exclude=.git src
endif
ifdef CSCOPE
	cscope -Rb
endif

#Make the Directories
directories:
	@mkdir -p $(BUILDDIR_S)
	@mkdir -p $(BUILDDIR_P)
	@mkdir -p $(BUILDDIR_UI)
	@mkdir -p $(BUILDDIR_LIBS)

#Clean only Objects
clean:
	@$(RM) -rf $(BUILDDIR_S)
	@$(RM) -rf $(BUILDDIR_P)
	@$(RM) -rf $(BUILDDIR_UI)
	@$(RM) -rf $(BUILDDIR_LIBS)

#Pull in dependency info for *existing* .o files
-include $(OBJECTS_S:.$(OBJEXT)=.$(DEPEXT))

#Link
$(TARGET_S): $(UILIBPATH) $(OBJECTS_S) $(OBJECTS_LIBS)
	$(CC) $(CCFLAGS) -o $(BIN_DIR)/$(TARGET_S) $^ $(UILIBPATH) $(LIBS)

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

parallel: directories libs $(UILIBPATH) $(OBJECTS_P) $(TARGET_P) $(OBJECTS_LIBS) ctags_p
	$(NVCC) -Xcompiler -fPIC --device-link $(NVCCLINK) -o $(BUILDDIR_LINK)/link.o $(OBJECTS_P) $(OBJECTS_LIBS) $(UILIBPATH) $(LIBS) -lcudadevrt -lcudart
	$(CC) $(CCFLAGS) -shared -o $(LIBRARY_DIR)/$(LIBRARY) $(OBJECTS_P) $(OBJECTS_LIBS) $(BUILDDIR_LINK)/link.o $(UILIBPATH) $(LIBS) -L/usr/local/cuda-8.0/lib64 -lcuda -lcudadevrt -lcudart

ctags_p: $(OBJECTS_P)
ifdef CTAGS
	ctags -R --exclude=.git src
endif
ifdef CSCOPE
	cscope -Rb
endif

#Pull in dependency info for *existing* .o files
-include $(OBJECTS_P:.$(OBJEXT)=.$(DEPEXT))

$(TARGET_P): $(UILIBPATH) $(OBJECTS_P) $(OBJECTS_LIBS)
	$(NVCC) $(NVCCLINK) -o $(BIN_DIR)/$(TARGET_P) $^ $(UILIBPATH) $(LIBS)

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
