#include <variant>
#include <concepts>
#include <string>
#include <cstddef>

class Derived {
public:
    void PrintName() const { 
        std::cout << "calling Derived!\n";
    }
    template <typename T>
    auto add_two(T a, T b) const {
        return a+b;
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
    template <typename T>
    auto add_two(T a, T b) const {
        return a-b;
    }
};

class One{
public:
    void print_hi(){
        std::cout << "hii\n";
    }
};

struct CallPrintName {
    void operator()(const Derived& d) { d.PrintName(); }    
    void operator()(const ExtraDerived& ed) { ed.PrintName(); }    
    void operator()(const One& o) {std::cout << "called bad\n";}
};

template <typename T>
struct CallAddTwo{
    T operator()(const Derived& d) { return d.add_two(a,b); }    
    T operator()(const ExtraDerived& ed) { return ed.add_two(a,b); }  
    T operator()(const One& o) {return 1;}  
    const T& a;
    const T& b; 
};

//could work
//get weight would need to pass a ref to a dummy matrix
//okay inputs can be different type put output type must be fixed

void test_variant(){
    std::vector<std::variant<Derived, ExtraDerived, One>> variant_vec;
    variant_vec.emplace_back(Derived{});
    variant_vec.emplace_back(ExtraDerived{});
    variant_vec.emplace_back(One{});
    for (auto &var: variant_vec){
        std::cout << std::visit(CallAddTwo<decltype(1.01)>{1.01,2.01}, var) << std::endl;

    }
    std::cout << 1.0 << std::endl;
    std::cout << 1.03 << std::endl;
    for (auto &var: variant_vec){
        std::cout << std::visit(CallAddTwo<decltype(1)>{1,2}, var) << std::endl;

    }
    
    std::visit(CallPrintName{}, variant_vec[1]);

}



