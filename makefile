#Compiler and Linker
CC          := g++
NVCC        := nvcc

#The Target Binary Program
TARGET_S      := test
TARGET_P      := parallel_test

#The Directories, Source, Includes, Objects, Binary and Resources
SRCDIR      := src
BUILDDIR_S  := ./build/serial
BUILDDIR_P  := ./build/parallel
TARGETDIR   := .
SRCEXT      := cpp
DEPEXT      := d
OBJEXT      := o

#Flags, Libraries and Includes
CCFLAGS      := -w -std=c++11 -pthread -Isrc
NVCCFLAGS    := -w -std=c++11 -Wno-deprecated-gpu-targets -x cu -DPARALLEL -Isrc

#---------------------------------------------------------------------------------
#DO NOT EDIT BELOW THIS LINE
#---------------------------------------------------------------------------------
SOURCES       := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS_S     := $(patsubst $(SRCDIR)/%,$(BUILDDIR_S)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

#Default Make
all: serial

serial: $(TARGET_S)

#Make the Directories
directories:
	@mkdir -p $(BUILDDIR_S)
	@mkdir -p $(BUILDDIR_P)

#Clean only Objecst
clean:
	@$(RM) -rf $(BUILDDIR_S)
	@$(RM) -rf $(BUILDDIR_P)

#Pull in dependency info for *existing* .o files
-include $(OBJECTS_S:.$(OBJEXT)=.$(DEPEXT))

#Link
$(TARGET_S): $(OBJECTS_S)
	$(CC) $(CCFLAGS) -o $(TARGETDIR)/$(TARGET_S) $^

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

$(TARGET_P): $(OBJECTS_P)
	$(NVCC) -Wno-deprecated-gpu-targets -L/usr/local/cuda-8.0/lib64 -DPARALLEL -lcuda -lcudart -o $(TARGETDIR)/$(TARGET_P) $^

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
