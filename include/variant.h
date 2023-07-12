#include <variant>
#include "layers.h"

class Derived {
public:
    void PrintName() const { 
        std::cout << "calling Derived!\n";
    }
};

class ExtraDerived {
public:
    void PrintName() const { 
        std::cout << "calling ExtraDerived!\n";
    }
    void PrintHelloWorld(){
        std::cout << "hi\n";
    }
};

struct CallPrintName {
    void operator()(const Derived& d) { d.PrintName(); }    
    void operator()(const ExtraDerived& ed) { ed.PrintName(); }    
};


void test_variant(){
    std::vector<std::variant<Derived, ExtraDerived>> layers;
    layers.emplace_back(Derived);
    layers.emplace_back(ExtraDerived);
    for (&layer : layers){
        layer->PrintName()
    }
    layers[1]->PrintHelloWorld();
}



