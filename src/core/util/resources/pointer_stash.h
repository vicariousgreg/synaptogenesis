#ifndef pointer_stash_h
#define pointer_stash_h

#include <set>
#include <map>

#include "util/resources/pointer.h"

class State;

class PointerStash {
    public:
        static PointerStash *get_instance();
        virtual ~PointerStash();

        void add(State* state, std::map<PointerKey, BasePointer*> new_pointers);
        const std::map<PointerKey, BasePointer*> get(State* state=nullptr) const;
        BasePointer* get(PointerKey key) const;
        void clear(State* state=nullptr);

    private:
        static PointerStash *instance;
        PointerStash();

        std::map<State*, std::map<PointerKey, BasePointer*>> owners;
        std::map<PointerKey, BasePointer*> pointers;
};

#endif
